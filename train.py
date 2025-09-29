import argparse

from tqdm import tqdm
import math
import logging
import nvtx
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import HybridAdam

from utils import load_tokenized_dataset, set_seed
from comm import LoadBalanceAllreduce as LBA

access_token = "YOUR_HUGGINGFACE_ACCESS_TOKEN"
skip = True


def setup_logger():
    # Get the rank of the current process
    rank = dist.get_rank()

    # Create a logger
    logger = logging.getLogger(f"rank_{rank}_logger")
    logger.setLevel(logging.INFO)

    # Create a file handler that logs to a specific file for each rank
    log_file = f"{rank}.log"
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)

    # Create a terminal (stream) handler to log to stdout (terminal)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter(f'[rank{rank}]: %(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def compress_with_allgather(top_k, tensor, pg_group):
    numel = tensor.numel()
    if numel <= 4096:
        return tensor
    tensor_abs = tensor.abs()
    tensor_sample = tensor_abs[torch.randint(0, numel, (math.ceil(0.01*numel), ), device=tensor.device)]

    tensor_samples = torch.empty(pg_group.size()*tensor_sample.numel(), device=tensor.device, dtype=tensor_abs.dtype)   
    dist.all_gather_into_tensor(tensor_samples, tensor_sample, pg_group, async_op=True).wait()
    with nvtx.annotate("topk"):
        topk = torch.topk(tensor_samples, tensor_samples.numel()*top_k//100,0, largest=True, sorted=False)[0]
    with nvtx.annotate("min"):
        threshold = torch.min(topk)
    with nvtx.annotate("where"):
        # mask = torch.ge(tensor.abs(),threshold)
        # tensor = tensor * mask
        tensor = torch.where(tensor_abs >= threshold, tensor, torch.tensor(0, device=tensor.device))
    return tensor   

def compress(top_k, tensor):
    numel = tensor.numel()
    if numel <= 4096:
        return tensor
    tensor_abs = tensor.abs()
    tensor_sample = tensor_abs[torch.randint(0, numel, (math.ceil(0.01*numel), ), device=tensor.device)]

    topk = torch.topk(tensor_sample, tensor_sample.numel()*top_k//100,0, largest=True, sorted=False)[0]
    threshold = torch.min(topk)
    return torch.where(tensor_abs >= threshold, tensor, torch.tensor(0, device=tensor.device))

def main():

    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-2.7b", help="Model name")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-s", "--num_steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k filtering for gradient")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # Method: 0 is default allreduce, 1 is compress then allgather, 2 is compress then our method(without bitmap), 3 is compress then our method(with bitmap)
    parser.add_argument("--method", type=int, default=3, help="Sync method")
    parser.add_argument("--bucket_cap", type=int, default=512, help="DDP bucket cap in MB")
    parser.add_argument("--num_hidden_layers", type=int, default=None, help="Model num_hidden_layers")
    parser.add_argument("--kv_heads", type=int, default=None, help="Model kv_heads")
    args = parser.parse_args()

    set_seed(args.seed)

    colossalai.launch_from_torch()
    coordinator = DistCoordinator()

    logger = setup_logger()

    # ==============================
    # Initialize Booster
    # ==============================
    plugin = HybridParallelPlugin(
        tp_size=args.tp,
        pp_size=1,
        precision = "bf16",
        ddp_bucket_cap_mb=args.bucket_cap,
    )
    booster = Booster(plugin=plugin)
    
    # ==============================
    # Initialize Dataset and Dataloader
    # ==============================
    logger.info("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = load_tokenized_dataset(tokenizer, args.max_length)
    logger.info("Loading tokenizer and dataset... Done")
    train_dataset = tokenized_datasets
    dataloader = plugin.prepare_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, seed=args.seed)


    # ==============================
    # Initialize Model and Optimizer
    # ==============================
    init_ctx = LazyInitContext(default_device=get_accelerator().get_current_device())
    config = AutoConfig.from_pretrained(args.model_name, token=access_token, trust_remote_code=True)
    if args.num_hidden_layers != None:
        config.num_hidden_layers = args.num_hidden_layers
    if args.kv_heads != None:
        config.num_key_value_heads = args.kv_heads
    if "gemma" in args.model_name:
        config.num_key_value_heads = 8
    with init_ctx:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    model.config._attn_implementation = "eager"
    optimizer = HybridAdam(model.parameters())
    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

    # ==============================
    # register communication hook to replace the default allreduce with our method
    # ==============================
    dp_group = model.dp_group

    dp_global_ranks = plugin.pg_mesh.get_ranks_in_group(dp_group)
    stream = torch.cuda.Stream()
    load_balance_allreduce = LBA(dp_group, dp_global_ranks, logger)
    def custom_comm_hook(state: object, bucket: dist.GradBucket):
        tensors = bucket.buffer()
        future = torch.futures.Future()
        global skip
        if skip == True:
            future.set_result(tensors)
            skip = False
            return future
        with torch.cuda.stream(stream):
            with nvtx.annotate("compress"):
                tensors_compressed = compress_with_allgather(args.top_k, tensors, model.tp_group)
            if dp_group.size() > 1:
                tensors = load_balance_allreduce.zen_sparse_comm(tensors_compressed, args.method)

        future.set_result(tensors)

        return future
    if args.method != 0:
        model.module.register_comm_hook(None, custom_comm_hook)
    coordinator.print_on_master(
        f"Booster init max CUDA memory: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB"
    )

    # ==============================
    # Training Loop
    # ==============================
    for step, batch in enumerate(tqdm(dataloader, desc="Step", disable=not coordinator.is_master())):
        if step == 80:
            start_time = time.time()
        if step == 95:
            end_time = time.time()
        batch = {k: v.to(model.module.device) for k, v in batch.items()}
        with nvtx.annotate("forward"):
            outputs = model(**batch)
            loss = outputs[0]
            del outputs  # free memory
        coordinator.print_on_master(f"Step {step} loss: {loss}")
        with nvtx.annotate("backward"):
            booster.backward(loss, optimizer)
        # clear the cache if you have memory issues
        # torch.cuda.empty_cache()

        with nvtx.annotate("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad()
  
        if step >= args.num_steps:
            break

    coordinator.print_on_master(f"Max CUDA memory usage: {get_accelerator().max_memory_allocated()/1024**2:.2f} MB")
    # write the time to a file
    coordinator.print_on_master(f"{args.model_name} method {args.method} topk {args.top_k} dp {dp_group.size()}\n")
    coordinator.print_on_master(f"{(end_time - start_time)/15}\n")
    if coordinator.is_master():
        with open("time.txt", "a") as f:
            f.write(f"{args.model_name} method {args.method} topk {args.top_k} dp {dp_group.size()}\n")
            f.write(f"{(end_time - start_time)/15}\n")


if __name__ == "__main__":
    main()
