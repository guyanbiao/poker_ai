from train_poker_llm import PokerLLM

def train_model():
    poker_llm = PokerLLM()
    poker_llm.train(num_episodes=1000)  # Train for more episodes

def use_trained_model():
    # Load the trained model
    poker_llm = PokerLLM()
    poker_llm.load_model()
    
    # Create a game state
    state = {
        'hand': ['A♠', 'K♠'],
        'pot': 100,
        'current_bet': 20
    }
    
    # Get model's action
    action = poker_llm.get_action(state)
    print(f"Model's action: {action}")

def main():
    # Uncomment the function you want to use
    # train_model()  # To train the model
    use_trained_model()  # To use the trained model

if __name__ == "__main__":
    main() 