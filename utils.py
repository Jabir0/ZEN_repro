import os
import csv
import math
import torch
import torch.distributed as dist
import random
import numpy as np
import transformers
from datasets import load_dataset

def load_tokenized_dataset(tokenizer, MAX_LENGTH=512):
    #dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", trust_remote_code=True, num_proc=128)
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", trust_remote_code=True, num_proc=128, split="train")
    subset_ds = dataset.select(range(10000))
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
        labels = inputs["input_ids"].copy()
        # Shift the input_ids to create labels
        for i in range(len(labels)):
            if labels[i] is not None:
                labels[i] = labels[i][1:] + [tokenizer.pad_token_id]
        inputs["labels"] = labels
        return inputs

    # Tokenize the datasets
    tokenized_datasets = subset_ds.map(tokenize_function, batched=True, num_proc=128)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    if "meta" in tokenized_datasets.column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(["meta"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

class Logger:
    def __init__(self, model_name, batchsize, top_k, partition, tp=8, output_dir="results"):
        self.model_name = model_name
        self.batchsize = batchsize
        self.top_k = top_k
        self.partition = partition
        self.tp = tp
        self.output_dir = os.path.join(output_dir, model_name+"_"+str(partition))
        self._create_directories()
        self._initialize_logging_files()

    def _create_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_logging_files(self):
        with open(self._get_file_path('sparsity_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Layer', 'Gradient Sparsity'])
        
        with open(self._get_file_path('skewness.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Layer', 'Skewness', 'Size'])

    def _get_file_path(self, filename):
        return os.path.join(self.output_dir, f'batchsize_{self.batchsize}_top_k_{self.top_k}_{filename}')

    def log_skewness(self, grad, param_name, step):
        axis = 0
        if grad.dim() == 2:
            axis = 0 if grad.shape[0] >= grad.shape[1] else 1

        grad_sample = grad.view(-1)
        grad_sample = torch.sort(grad_sample.abs())[0]
        approx_index = int((1 - self.top_k * 0.01) * len(grad_sample))
        threshold = grad_sample[approx_index]

        grad = torch.where(grad.abs() >= threshold, grad, torch.tensor(0.0, dtype=grad.dtype))
        grad_parts = torch.chunk(grad, self.partition, dim=axis)
        total_non_zero = torch.count_nonzero(grad).cpu().numpy()
        if total_non_zero != 0:
            non_zero_ratios = [torch.count_nonzero(part).cpu().numpy() / total_non_zero for part in grad_parts]
        else:
            non_zero_ratios = [0 for part in grad_parts]
            non_zero_ratios[0] = 1.0
        non_zero_ratios = [float(val) for val in non_zero_ratios]
        with open(self._get_file_path('skewness.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([step, param_name, non_zero_ratios, grad.numel()])
        return grad
    
    def log_skewness_distributed(self, grad, param_name, step):
        numel = grad.numel()
        if numel <= 1500:
            return grad

        grad_abs = grad.view(-1).abs()
        grad_sample = grad_abs[torch.randint(0, numel, (math.ceil(0.01*numel), ), device=grad.device)]

        grad_samples = [torch.zeros_like(grad_sample) for _ in range(dist.get_world_size())]
        dist.all_gather(grad_samples, grad_sample)
        grad_samples = torch.stack(grad_samples).view(-1)

        topk = torch.topk(grad_samples, grad_samples.numel()*self.top_k//100,0, largest=True, sorted=False)[0]
        threshold = torch.min(topk)

        grad = torch.where(grad.abs() >= threshold, grad, torch.tensor(0.0, dtype=grad.dtype))
        total_non_zero = torch.count_nonzero(grad)
        total_non_zeros = [torch.zeros_like(total_non_zero) for _ in range(dist.get_world_size())]
        dist.all_gather(total_non_zeros, total_non_zero)

        # Convert list of tensors to a single tensor
        total_non_zeros_tensor = torch.stack(total_non_zeros)
        # Sum the tensor
        total_sum = torch.sum(total_non_zeros_tensor)

        axis = 0
        if grad.dim() == 2:
            axis = 0 if grad.shape[0] >= grad.shape[1] else 1
        grad_parts = torch.chunk(grad, self.partition, dim=axis)
        if total_sum != 0:
            non_zero_ratios = [torch.count_nonzero(part).item() / total_sum.item() for part in grad_parts]
        else:
            non_zero_ratios = [0 for part in grad_parts]
            non_zero_ratios[0] = 1.0

        non_zero_ratios_tensor = torch.tensor(non_zero_ratios, device=grad.device)
        gathered_non_zero_ratios = [torch.zeros_like(non_zero_ratios_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_non_zero_ratios, non_zero_ratios_tensor)
        non_zero_ratios = torch.cat(gathered_non_zero_ratios).view(-1)
        if dist.get_rank() == 0:
            with open(self._get_file_path('skewness.csv'), mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([step, param_name, non_zero_ratios.cpu().tolist(), grad.numel()*self.tp])
        return grad

    def log_sparsity(self, model, step):
        with open(self._get_file_path('sparsity_data.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            for name, param in model.named_parameters():
                    grad = param.grad
                    grad = self.log_skewness(grad, name, step)
                    param.grad = grad
                    grad_sparsity = self.calculate_sparsity(grad)
                    writer.writerow([step, name, grad_sparsity,])

    @staticmethod
    def calculate_sparsity(tensor, threshold=1e-5):
        num_elements = tensor.numel()
        num_near_zeros = (tensor == 0).sum().item()
        sparsity = num_near_zeros / num_elements
        return sparsity
