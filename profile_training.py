# profile_training.py
# Advanced training performance profiling with optimized data loading

import torch
import torch.optim as optim
from tqdm import tqdm

# Import model components and utilities
from Transformer_Interpretability import (
    TransformerModel, ExperimentConfig, MixedSequenceDataset, seq_collate_fn,
    compute_next_token_loss, tiktoken
)

def profile_run():
    """Sets up and runs a short, instrumented training loop for profiling."""
    
    # 1. Configuration
    print("Loading configuration...")
    cfg = ExperimentConfig()
    device = torch.device(cfg.device_id)
    encoder = tiktoken.get_encoding("gpt2")
    print(f"Running on device: {device}")

    # 2. Model Initialization
    print("Initializing model...")
    model = TransformerModel(
        encoder.n_vocab, cfg.embed_size, cfg.transformer_n_heads,
        cfg.transformer_n_blocks, cfg.block_size, cfg.dropout_rate
    ).to(device)

    # 3. Dummy Data & DataLoader
    # We create a synthetic dataset in memory for a quick and repeatable profiling run.
    print("Creating dummy dataset...")
    num_sequences = 256
    dummy_sequences = [
        torch.randint(0, encoder.n_vocab, (cfg.block_size,)).tolist()
        for _ in range(num_sequences)
    ]
    dataset = MixedSequenceDataset(dummy_sequences, [], 1.0)
    
    # Initialize optimized DataLoader with enhanced performance settings
    print("Initializing optimized DataLoader with parallel processing...")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,        # Parallel data loading processes
        collate_fn=seq_collate_fn,
        pin_memory=True       # Asynchronous GPU memory transfer
    )
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.base_learning_rate)

    # Execute optimized training loop for performance profiling
    print("Starting profiling run for 100 steps...")
    model.train()
    max_steps = 100
    step_count = 0
    
    # Main training loop with performance optimizations
    while step_count < max_steps:
        for batch in loader:
            if step_count >= max_steps:
                break
            
            optimizer.zero_grad()
            # Asynchronous GPU transfer for optimal performance
            batch_gpu = batch.transpose(0, 1).to(device, non_blocking=True)
            
            logits = model(batch_gpu)
            loss = compute_next_token_loss(logits, batch_gpu)
            
            loss.backward()
            optimizer.step()
            
            step_count += 1
            if step_count % 10 == 0:
                print(f"  Step {step_count}/{max_steps}, Loss: {loss.item():.4f}")

    print("✅ Profiling run finished.")

if __name__ == "__main__":
    """
    Main execution for training performance profiling.
    
    To run with Nsight Systems profiling:
    !nsys profile -t cuda,nvtx,osrt -o optimized_profile --force-overwrite true python profile_training.py
    
    This will generate 'optimized_profile.nsys-rep' for detailed performance analysis.
    """
    profile_run()
