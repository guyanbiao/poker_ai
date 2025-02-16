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
    
    num_episodes = 5  # Increased for better learning
    
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

def analyze_strategy(poker_llm):
    """Analyze the model's strategy for each possible hand"""
    hands = [
        ['A♠', 'A♣'],  # AA
        ['K♠', 'K♣'],  # KK
        ['Q♠', 'Q♣']   # QQ
    ]
    
    bet_sizes = [10, 30, 50]  # Test different bet sizes
    print("\nSTRATEGY ANALYSIS")
    print("=" * 50)
    
    for hand in hands:
        print(f"\nHand: {' '.join(hand)} (Strength: {poker_llm.env._calculate_hand_strength(hand):.1f})")
        print("-" * 30)
        actions = {'fold': 0, 'call': 0, 'raise': 0}
        
        for bet in bet_sizes:
            for _ in range(5):  # Test each scenario multiple times
                state = {
                    'hand': hand,
                    'opponent_hand': None,  # Not needed for decision
                    'pot': 100,
                    'current_bet': bet,
                    'current_player': 0
                }
                action = poker_llm.get_action(state)
                actions[action] += 1
        
        total = sum(actions.values())
        print(f"Action distribution across different bet sizes:")
        for action, count in actions.items():
            percentage = (count / total) * 100
            print(f"{action}: {percentage:.1f}%")

def main():
    print("Training new model...")
    poker_llm = train_model()
    
    # Run strategy analysis
    analyze_strategy(poker_llm)
    
    print("\nTesting specific matchups...")
    test_hands = [
        (['A♠', 'A♣'], ['K♠', 'K♣']),  # AA vs KK (should raise)
        (['K♠', 'K♣'], ['Q♠', 'Q♣']),  # KK vs QQ (should raise)
        (['Q♠', 'Q♣'], ['A♠', 'A♣']),  # QQ vs AA (should fold)
        (['K♠', 'K♣'], ['A♠', 'A♣']),  # KK vs AA (should fold/call)
        (['Q♠', 'Q♣'], ['K♠', 'K♣'])   # QQ vs KK (should fold/call)
    ]
    
    print("\nSPECIFIC MATCHUP RESULTS")
    print("=" * 50)
    
    for player_hand, opponent_hand in test_hands:
        state = {
            'hand': player_hand,
            'opponent_hand': opponent_hand,
            'pot': 100,
            'current_bet': 20,
            'current_player': 0
        }
        action = poker_llm.get_action(state)
        print(f"\nMatchup: {' '.join(player_hand)} vs {' '.join(opponent_hand)}")
        print(f"Model's action: {action}")
        print(f"Expected action: {'raise' if player_hand[0][0] > opponent_hand[0][0] else 'fold/call'}")
        print(f"Correct? {'✓' if (player_hand[0][0] > opponent_hand[0][0] and action == 'raise') or (player_hand[0][0] < opponent_hand[0][0] and action in ['fold', 'call']) else '✗'}")

if __name__ == "__main__":
    main() 