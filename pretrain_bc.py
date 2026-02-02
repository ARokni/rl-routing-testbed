
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from grid_routing_env import GridRoutingEnv
from BackPressure import BackPressureExpert
from Actor_Critic import ActorCritic

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_expert_data(env, expert, samples=10000):
    """Generates a dataset of (observation, expert_action) pairs."""
    print(f"Generating {samples} expert samples...")
    obs_list = []
    act_list = []
    
    obs = env.reset()
    for _ in range(samples):
        # 1. Record Observation
        obs_list.append(obs)
        
        # 2. Get Expert Action
        # Expert returns list [a0, a1, a2]
        action = expert.act(obs)
        act_list.append(action)
        
        # 3. Step Environment (to keep distribution valid)
        obs, _, done, _ = env.step(action)
        if done: # Should not happen often in this infinite horizon env, but good practice
            obs = env.reset()
            
    return np.array(obs_list), np.array(act_list)

def main():
    # --- Hyperparameters ---
    BATCH_SIZE = 128
    LR = 3e-3
    EPOCHS = 1
    N_SAMPLES = 1000  # Size of dataset
    MODEL_SAVE_PATH = "bc_model_new.pt"

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridRoutingEnv()
    expert = BackPressureExpert(env)
    
    # 2. Generate Data
    X_train, y_train = generate_expert_data(env, expert, samples=N_SAMPLES)
    
    # Convert to Tensors
    tensor_x = torch.Tensor(X_train).to(device)
    tensor_y = torch.LongTensor(y_train).to(device) # Actions are integers
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Model
    model = ActorCritic(obs_dim=env.observation_space.shape[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training Loop
    print(f"Starting BC Training on {device}...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass (returns tuple of logits for the 3 nodes)
            (logits0, logits1, logits2), _ = model(batch_x)
            
            # Calculate Loss for each node (Multi-Head Classification)
            # batch_y shape is [batch, 3], so we slice column-wise
            l0 = loss_fn(logits0, batch_y[:, 0])
            l1 = loss_fn(logits1, batch_y[:, 1])
            l2 = loss_fn(logits2, batch_y[:, 2])
            
            loss = l0 + l1 + l2
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 1 == 0:
          with torch.no_grad():
              logits,_ = model(tensor_x)
              preds = torch.stack([l.argmax(-1) for l in logits], dim=1)
              acc = (preds == tensor_y).float().mean().item()
          print(f"[BC] epoch {epoch+1:02d}  acc={acc:.3f}")
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 5. Save
    # We save only the state dict to keep it clean
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training Complete. Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
