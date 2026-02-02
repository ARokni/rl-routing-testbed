# RL Routing - Real World Fine-Tuning

This repository contains the PPO fine-tuning module for the routing testbed.

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`

## Usage
1. Place the `bc_model.pt` in the root directory.
2. Run the fine-tuning script:
   `python train_online.py`

## Integration
To connect this to the real-world testbed, modify the environment initialization in `train_online.py` to use the hardware interface wrapper instead of `GridRoutingEnv`.
