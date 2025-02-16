from train_poker_llm import PokerLLM
import os

def train_model():
    poker_llm = PokerLLM()
    poker_llm.train(num_episodes=20)  # Train for more episodes

def use_trained_model():
    # Check if model exists locally
    if not os.path.exists("poker_model"):
        print("No trained model found. Training a new model first...")
        train_model()
    
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
    # First train the model
    print("Training new model...")
    train_model()
    
    # Then use it
    print("\nTesting trained model...")
    use_trained_model()

if __name__ == "__main__":
    main() 