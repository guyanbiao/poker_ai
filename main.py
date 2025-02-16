from train_poker_llm import PokerLLM
import os
import matplotlib.pyplot as plt
import numpy as np

def train_model():
    # Check if we have a saved model
    if os.path.exists("poker_model"):
        print("Loading existing model...")
        poker_llm = PokerLLM()
        poker_llm.load_model("poker_model")
    else:
        print("Starting with fresh model...")
        poker_llm = PokerLLM()
    
    num_episodes = 30  # Increased for better learning
    
    # Track metrics
    rewards_history = []
    losses_history = []
    actions_history = {'fold': 0, 'call': 0, 'raise': 0}
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = poker_llm.env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0
        
        while not done:
            action = poker_llm.get_action(state)
            next_state, reward, done = poker_llm.env.step(action)
            episode_reward += reward
            
            # Track action frequencies
            actions_history[action] += 1
            
            state = next_state
        
        rewards_history.append(episode_reward)
        losses_history.append(episode_loss)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}")
            print(f"Reward: {episode_reward:.3f}")
            print(f"Current action distribution:")
            total_actions = sum(actions_history.values())
            for action, count in actions_history.items():
                percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                print(f"{action}: {percentage:.1f}%")
            print("------------------------")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    actions = list(actions_history.keys())
    counts = list(actions_history.values())
    plt.bar(actions, counts)
    plt.title('Action Distribution')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print(f"Training results saved to training_results.png")
    
    return poker_llm

def main():
    print("Training new model...")
    poker_llm = train_model()
    
    print("\nTesting trained model...")
    # Test with all possible hand combinations
    test_hands = [
        (['A♠', 'A♣'], ['K♠', 'K♣']),  # AA vs KK (should raise)
        (['K♠', 'K♣'], ['Q♠', 'Q♣']),  # KK vs QQ (should raise)
        (['Q♠', 'Q♣'], ['A♠', 'A♣']),  # QQ vs AA (should fold)
        (['K♠', 'K♣'], ['A♠', 'A♣']),  # KK vs AA (should fold/call)
        (['Q♠', 'Q♣'], ['K♠', 'K♣'])   # QQ vs KK (should fold/call)
    ]
    
    for player_hand, opponent_hand in test_hands:
        state = {
            'hand': player_hand,
            'opponent_hand': opponent_hand,
            'pot': 100,
            'current_bet': 20,
            'current_player': 0
        }
        action = poker_llm.get_action(state)
        print(f"\nTest case:")
        print(f"Your hand: {player_hand}")
        print(f"Opponent's hand: {opponent_hand}")
        print(f"Model's action: {action}")
        print(f"Expected action: {'raise' if player_hand[0][0] > opponent_hand[0][0] else 'fold/call'}")

if __name__ == "__main__":
    main() 