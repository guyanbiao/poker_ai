import os
import sys
import time
from poker.training.environment import PokerEnvironment
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def download_model(model_name: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
    """Download the model if it doesn't exist."""
    import urllib.request
    import hashlib
    
    models_dir = "models"
    model_path = os.path.join(models_dir, model_name)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    # Download the model
    print(f"Downloading {model_name}...")
    url = f"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/{model_name}"
    
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            def show_progress(block_num, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)
            urllib.request.urlretrieve(url, model_path, show_progress)
        print(f"Model downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def plot_results(reward_history):
    """Plot the cumulative rewards for each player over time."""
    episodes = len(reward_history)
    num_players = len(reward_history[0])
    
    # Calculate cumulative rewards
    cumulative_rewards = np.zeros((episodes, num_players))
    for i in range(episodes):
        for player in range(num_players):
            if i == 0:
                cumulative_rewards[i, player] = reward_history[i][player]
            else:
                cumulative_rewards[i, player] = cumulative_rewards[i-1, player] + reward_history[i][player]
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for player in range(num_players):
        plt.plot(cumulative_rewards[:, player], label=f'Player {player}')
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Poker Agent Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.close()

def main():
    start_time = time.time()
    
    # Download or verify model exists
    print("Step 1: Setting up model...")
    model_path = download_model()
    
    # Initialize environment
    print("\nStep 2: Initializing environment...")
    num_players = 3
    starting_stack = 1000
    small_blind = 5
    num_episodes = 3
    
    env = PokerEnvironment(num_players, starting_stack, small_blind)
    
    # Run training session
    print(f"\nStep 3: Running {num_episodes} episodes...")
    reward_history = env.run_training_session(num_episodes)
    
    # Plot and save results
    print("\nStep 4: Plotting results...")
    plot_results(reward_history)
    
    # Print final statistics
    print("\nFinal Statistics:")
    total_rewards = {i: sum(episode[i] for episode in reward_history) 
                    for i in range(num_players)}
    
    for player, reward in total_rewards.items():
        print(f"Player {player}: Total Reward = {reward}, Average Reward per Episode = {reward/num_episodes:.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Average time per episode: {total_time/num_episodes:.2f} seconds")

if __name__ == "__main__":
    main() 