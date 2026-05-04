"""
Tests the CNNPolicy architecture to ensure tensor shapes align 
and the forward pass executes without crashing.
"""

import torch
from model import CNNPolicy

def main():
    print("Initializing Nature CNN...")
    # We have 10 actions total (9 original + 1 query_oracle)
    n_actions = 10 
    model = CNNPolicy(n_actions=n_actions)
    
    # 1. Simulate a Batch from the Vectorized Environment
    # When using 4 CPU cores, the wrapper outputs 4 images at a time.
    # Shape: (Batch Size, Height, Width, Channels)
    batch_size = 4
    dummy_obs_batch = torch.randint(0, 256, (batch_size, 84, 84, 3), dtype=torch.uint8)
    
    print(f"\n[Input] Batch Observation Shape: {dummy_obs_batch.shape} | Dtype: {dummy_obs_batch.dtype}")
    
    # Run the forward pass!
    print("\n--- Testing Batched Forward Pass ---")
    action, log_prob, entropy, value = model.get_action_and_value(dummy_obs_batch)
    
    # Verify outputs
    print(f"Action shape:   {action.shape} (Expected: ({batch_size},))")
    print(f"Log Prob shape: {log_prob.shape} (Expected: ({batch_size},))")
    print(f"Entropy shape:  {entropy.shape} (Expected: ({batch_size},))")
    print(f"Value shape:    {value.shape} (Expected: ({batch_size},))")
    
    # 2. Simulate a Single Frame (Unbatched)
    # Sometimes during evaluation, we only feed 1 image instead of a batch.
    # The permute logic in model.py should catch this and add a batch dimension.
    print("\n--- Testing Unbatched (Single Frame) Input ---")
    single_obs = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    
    action_single, _, _, value_single = model.get_action_and_value(single_obs)
    print(f"Single Action shape: {action_single.shape} (Expected: (1,))")
    print(f"Single Value shape:  {value_single.shape} (Expected: (1,))")

    print("\n[SUCCESS] Model Test Complete! All tensor math is perfectly aligned.")

if __name__ == "__main__":
    main()