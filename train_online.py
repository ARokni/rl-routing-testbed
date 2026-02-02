import torch
import copy
import os
import sys

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from grid_routing_env import GridRoutingEnv
from Actor_Critic import ActorCritic, PPOFineTuner

def main():
    print("--- Setting up Online Fine-Tuning ---")

    # 1. Initialize the Environment
    # NOTE: Your friends can replace 'GridRoutingEnv' here with their real-world testbed wrapper.
    # The wrapper must adhere to the standard Gym interface (reset, step).
    env = GridRoutingEnv(beta=1.0, lambda_p=0.0, mu_q=0.1) 
    obs_dim = env.observation_space.shape[0]

    # 2. Initialize the Model Architecture
    model = ActorCritic(obs_dim)

    # 3. Load the Pre-Trained Weights (Behavioral Cloning)
    model_path = "bc_model.pt"
    if os.path.exists(model_path):
        print(f"Loading pre-trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {model_path} not found. Starting from scratch.")

    # 4. Setup PPO Fine-Tuner
    # Using the hyperparameters from your notebook
    ppo = PPOFineTuner(
        model,
        vf_coef=1,
        minibatch_size=256,
        lr=3e-4,
        actor_epochs=4,
        critic_epochs=8,
        beta_kl=0.2,
        kl_targ=0.9,
        clip=0.2,
        ent_coef=0.01,
        threshold=20,
        policy_coeff=1
    )

    # 5. Start Training
    print("Starting PPO fine-tuning...")
    # You can adjust epochs or steps_per_epoch as needed for the testbed
    ppo.train(env, epochs=30, steps_per_epoch=1024)
    
    # 6. Save the Fine-Tuned Model
    torch.save(ppo.net.state_dict(), "finetuned_model.pt")
    print("Training complete. Model saved to 'finetuned_model.pt'.")

if __name__ == "__main__":
    main()
