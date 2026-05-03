"""
Benchmarks the BFS Oracle on gym-sokoban.
Calculates average solve time and effective Steps-Per-Second (SPS).
"""

import gym
import gym_sokoban
import time
import numpy as np
from bfs_oracle import extract_state, bfs_sokoban

# Monkeypatch for NumPy 2.0 vs Old Gym
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def run_benchmark(episodes=100):
    env = gym.make('Sokoban-small-v0')
    
    solve_times = []
    total_steps_cached = 0
    failures = 0
    
    print(f"Starting Benchmark: 7x7 Sokoban, {episodes} episodes...")
    
    t_start_total = time.time()
    
    for ep in range(episodes):
        env.reset()
        unwrapped_env = env.unwrapped
        
        # Time the raw BFS computation
        t0 = time.time()
        path = bfs_sokoban(unwrapped_env, debug=False)
        t_solve = time.time() - t0
        
        if path is not None:
            solve_times.append(t_solve)
            total_steps_cached += len(path)
        else:
            failures += 1
            
        if (ep + 1) % 20 == 0:
            print(f"  Processed {ep + 1}/{episodes} maps...")
            
    total_time = time.time() - t_start_total
    
    if len(solve_times) == 0:
        print("Benchmark failed: No maps solved.")
        return
        
    avg_solve = np.mean(solve_times)
    max_solve = np.max(solve_times)
    
    # Calculate Effective SPS (Assuming Cache Hits)
    # The time it takes to pop from a cache is effectively 0.00001s
    effective_sps = total_steps_cached / total_time
    
    print("\n" + "="*40)
    print(" BENCHMARK RESULTS (7x7 BFS Oracle)")
    print("="*40)
    print(f"Total Maps Solved : {len(solve_times)}")
    print(f"Total Failures    : {failures} (Unsolvable generation)")
    print(f"Total Valid Steps : {total_steps_cached}")
    print("-"*40)
    print(f"Avg Solve Time    : {avg_solve:.4f} seconds")
    print(f"Max Solve Time    : {max_solve:.4f} seconds")
    print(f"Effective SPS     : {effective_sps:.0f} steps / second")
    print("="*40)
    
if __name__ == "__main__":
    run_benchmark(100)