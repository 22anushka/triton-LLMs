"Benchmarking transformer models"

#TODO
# initialize model
# random batch of data
# benchmark

import timeit
from .cs336_basics import transformer_model, data, optimizer, nn_utils

import argparse
import torch 
import numpy as np 
import pandas as pd 

if __name__=="__main__":
    
    
    parser = argparse.ArgumentParser(description="Transformer inputs")
    parser.add_argument("--context_len", default=4000, type=int, help="Context length for the model")
    parser.add_argument("--num_l", default=12, type=int, help="Number of layers for the transformer")
    parser.add_argument("--num_h", default=12, type=int, help="Number of heads for MHA")
    parser.add_argument("--theta", default=10000, type=int, help="Theta for rope")
    parser.add_argument("--d_model", default=768, type=int, help="Dimension of d_model (embedding dim)")
    parser.add_argument("--d_ff", default=3072, type=int, help="Up_projection dimension for MLP in transformer")
    parser.add_argument("--fb", default=0, type=int, help="Forward only:0, Forward and backward: 1 for benchmarking")
    args = parser.parse_args()
    
    fb = args.fb
    vocab_size = 10000
    batch_size = 4
    context_length = args.context_length # get from argument
    n_layers = args.n_layers
    n_heads = args.n_heads
    d_model = args.d_model
    d_ff = args.d_ff
    theta = args.theta # usual theta set for rope
    
    warmup_steps = 5 # let there be 5 warm up steps
    model = transformer_model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
    )
    
    if fb:
        optim = optimizer.Adam
    
    # generate a random batch of data
    # use torch.randint
    input_ids = torch.full((batch_size, context_length), 0, dtype=torch.long)
    for i in range(batch_size):
        current_seq_len = np.random.randint(1, context_length + 1)
        input_ids[i, :current_seq_len] = torch.randint(1, vocab_size, (current_seq_len,), dtype=torch.long)

    # run w warmup steps
    if fb:
        model.train()
    else:
        model.eval()
    
    def train(input_ids):
        optimizer.zero_grad()
        out = model.forward(input_ids).logits
        loss = nn_utils.cross_entropy(input_ids, out)
        loss.backward()
        optimizer.step()
    
    def run_code(steps):
        if fb:
            with torch.enable_grad():
                for i in range(steps):
                    train()
                    torch.cuda.synchronize()

        else:
            with torch.no_grad():
                for i in range(steps):
                    model.forward(input_ids)
                    torch.cuda.synchronize()

    # warmup
    print("Warm-up steps")
    run_code(warmup_steps)
    
    # benchmarking step
    print("Starting benchmark...")
    benchmark_steps = 2
    start_time = timeit.default_timer()
    
    run_code(benchmark_steps)
    
    end_time = timeit.default_timer()
    
                
            
        
    
    
    
    