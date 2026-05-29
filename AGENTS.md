# Upper Bound: Oracle-Guided PPO in MiniGrid

## Overview
This codebase implements an "Upper Bound" Reinforcement Learning experiment using Proximal Policy Optimization (PPO) augmented with an algorithmic **Oracle**. The agent operates in pixel-based MiniGrid environments and has access to a special `query_oracle` action. When triggered, the oracle calculates the optimal action using Breadth-First Search (BFS) and executes it, effectively providing expert guidance. The agent is trained using a combination of standard PPO loss and Behavior Cloning (BC) loss on the guided steps.

## Core Components

### 1. Environment Wrapper (`env_wrapper.py`)
- **Action Space Extension:** Modifies the standard MiniGrid action space by adding one extra action (`N`) mapped to `query_oracle`.
- **Observation:** Applies `FullyObsWrapper` (full grid visibility) and `RGBImgObsWrapper` (RGB image observation, typically 40x40 pixels for a tile size of 8).
- **Oracle Integration:** If the agent selects `query_oracle`, the wrapper intercepts this, consults the BFS algorithm to find the optimal action, executes the optimal action instead, and marks the transition as `guided=True`.
- **Cost Mechanism:** Applies an optional `oracle_cost` reward penalty to discourage over-reliance on the oracle.

### 2. The Oracle (`oracle.py`)
- Provides algorithmic experts for specific MiniGrid environments:
  - **`bfs_empty`**: Solves `MiniGrid-Empty` by finding the shortest path to the goal cell using BFS.
  - **`bfs_doorkey`**: Solves `MiniGrid-DoorKey` by navigating to the key, picking it up, navigating to the door, opening it, and finally moving to the goal.
- It returns the exact action index required (e.g., turn left, move forward, pickup, toggle).

### 3. Neural Network Policy (`model.py`)
- **Architecture:** A Convolutional Neural Network (CNN) backbone inspired by SAGE/Standard RL vision papers, designed to process (B, 3, H, W) float32 image tensors.
- **Layers:** Three consecutive `Conv2d` layers with ReLU activations, followed by a flattened fully connected layer.
- **Heads:** Splits into a categorical `policy_head` (action logits) and a scalar `value_head` (state value estimate).

### 4. Training Loop (`train.py`)
- **PPO Implementation:** A single-file implementation of PPO with Generalized Advantage Estimation (GAE).
- **Behavior Cloning (BC):** During PPO updates, if a step was guided by the oracle, an additional Behavior Cloning loss is applied (`-log_prob` of the oracle's chosen action) to teach the policy network to mimic the oracle.
- **Metrics Logging:** Tracks and logs `ep_return`, `success`, `guided_pct` (how often the oracle was used), `queries_per_ep`, `bc_loss`, and `agreement_rate` to CSV files.

### 5. Experiment Management & Visualization
- **Bash Scripts:** `job.sh`, `launch_all.sh`, `submit_all.sh`, and `run_all_sequential.sh` handle running multiple seeds and oracle cost configurations (free vs. paid) on SLURM clusters.
- **Plotting (`plot.py`, `compare_plot.py`):** Scripts to read the generated CSV logs and plot smoothed curves for Episodic Return, Success Rate, Oracle Guidance Usage, and Sample Efficiency (Return vs. Cumulative Oracle Calls).

## Agent Interface / Skills Definition
If this codebase is used as a tool/skillset for an AI agent, the agent should understand:
1. **How to Launch:** `python train.py --env-id <ENV> --env-type <empty|doorkey> --oracle-cost <FLOAT>`
2. **How to Modify Models:** Architecture changes go in `model.py`.
3. **How to Add Environments:** To support a new MiniGrid layout, a new solver must be added to `oracle.py` and linked inside `env_wrapper.py`.