#!/bin/bash

# Configurations
hostfiles=("hostfile_kaggle")
models=("google/gemma-2-2b:8:28" "meta-llama/Llama-3.2-3B:8:28" "facebook/opt-2.7b:8:12")

# Method: 0 is default allreduce, 1 is compress then allgather, 2 is compress then our method (without bitmap), 3 is compress then our method (with bitmap)
methods=(3 2 0) 

top_k_values=(10 5 1)

# Iterate over configurations in the new order: model -> top_k -> hostfile -> method
for model_config in "${models[@]}"; do
    IFS=':' read -r model_name tp num_hidden_layers <<< "$model_config"
    for top_k in "${top_k_values[@]}"; do
        for hostfile in "${hostfiles[@]}"; do
            for method in "${methods[@]}"; do
                echo "Running: $model_name on $hostfile with top_k=$top_k and method $method"
                colossalai run --nproc_per_node 1 --hostfile "$hostfile" \
                    train.py --model_name "$model_name" \
                    --grad_checkpoint --tp 1 --method "$method" \
                    --num_hidden_layers "$num_hidden_layers" --bucket_cap 128 \
                    --top_k "$top_k"
            done
        done
    done
done