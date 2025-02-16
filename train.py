import os
import sys
from poker.training.environment import PokerEnvironment
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def download_model(model_name: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
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
    url = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/{model_name}"
    
    try:
        urllib.request.urlretrieve(url, model_path)
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
    # Download or verify model exists
    model_path = download_model()
    
    # Initialize environment
    num_players = 4
    starting_stack = 1000
    small_blind = 5
    num_episodes = 3
    
    env = PokerEnvironment(num_players, starting_stack, small_blind)
    
    # Run training session
    print(f"Running {num_episodes} episodes...")
    reward_history = env.run_training_session(num_episodes)
    
    # Plot and save results
    plot_results(reward_history)
    
    # Print final statistics
    print("\nFinal Statistics:")
    total_rewards = {i: sum(episode[i] for episode in reward_history) 
                    for i in range(num_players)}
    
    for player, reward in total_rewards.items():
        print(f"Player {player}: Total Reward = {reward}, Average Reward per Episode = {reward/num_episodes:.2f}")

if __name__ == "__main__":
    main() 