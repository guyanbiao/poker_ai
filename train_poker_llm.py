import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Dict
import numpy as np
from poker_env import PokerEnvironment

import os
os.environ['USE_TORCH'] = '1'
os.environ['FORCE_CPU'] = '1'  # Force CPU usage
os.environ['NO_TENSORFLOW'] = '1'

class PokerLLM:
    def __init__(self, model_name: str = "distilgpt2"):
        # Force CPU usage
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize environment first
        self.env = PokerEnvironment()
        
        # Try to load existing model first
        if os.path.exists("poker_model"):
            try:
                self.load_model("poker_model")
                return
            except Exception as e:
                print(f"Could not load saved model: {e}")
                print("Starting with fresh model...")
        
        # Load fresh model if no saved model exists
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=False
        )
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32,
            local_files_only=False
        ).to(self.device)
        
    def format_state(self, state: Dict) -> str:
        """Convert poker state to text format for LLM"""
        hand_strength = self.env._calculate_hand_strength(state['hand'])
        return f"""
        You are playing poker with pocket pairs only (AA, KK, or QQ).
        Your hand: {' '.join(state['hand'])} (Hand strength: {hand_strength:.1f})
        Pot: ${state['pot']}
        Current bet: ${state['current_bet']}
        
        Strategy guide:
        - With AA (strength 1.0): Always raise
        - With KK (strength 0.8): Call or raise
        - With QQ (strength 0.6): Call if bet is small, fold if raised
        
        What action would you like to take? Choose one: fold/call/raise
        """
    
    def get_action(self, state: Dict) -> str:
        """Get model's action prediction"""
        input_text = self.format_state(state)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        try:
            input_length = inputs.input_ids.shape[1]
            max_length = input_length + 20  # Allow 20 more tokens for the response
            
            print(f"\nCurrent state:\n{input_text}")  # Print the current state
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,  # Use dynamic max_length
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            
            action = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_action = self._parse_action(action)
            print(f"Model output: {action}")
            print(f"Parsed action: {parsed_action}")
            return parsed_action
            
        except Exception as e:
            print(f"Error generating action: {e}")
            return 'fold'  # Default to fold if there's an error
    
    def train(self, num_episodes: int = 1000):
        """Train the model using reinforcement learning"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        # Load or initialize training history
        if not hasattr(self, 'training_history'):
            self.training_history = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_loss = 0
            
            while not done:
                # Get model's action prediction
                input_text = self.format_state(state)
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Take action in environment
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                episode_loss += loss.item()
                
                state = next_state
            
            # Save episode results
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'loss': episode_loss
            })
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss:.4f}")
        
        # Save the trained model and history
        self.save_model()

    def _parse_action(self, action_text: str) -> str:
        """Parse model output to valid poker action"""
        valid_actions = ['fold', 'call', 'raise']
        
        # If no valid action is found in the text, make a strategic choice
        # based on the current hand strength
        if not any(action in action_text.lower() for action in valid_actions):
            hand_strength = self.env._calculate_hand_strength(self.env.players[self.env.current_player])
            if hand_strength > 0.7:
                return 'raise'
            elif hand_strength > 0.4:
                return 'call'
            return 'fold'
            
        # Count occurrences of each action word
        action_counts = {
            action: action_text.lower().count(action)
            for action in valid_actions
        }
        
        # Return the most mentioned action
        if any(action_counts.values()):
            return max(action_counts.items(), key=lambda x: x[1])[0]
            
        return 'call'  # Default to call instead of fold

    def save_model(self, path: str = "poker_model"):
        """Save the trained model, tokenizer, and training history"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training metadata
        if hasattr(self, 'training_history'):
            np.save(os.path.join(path, 'training_history.npy'), self.training_history)
        
        print(f"Model and history saved to {path}")
    
    def load_model(self, path: str = "poker_model"):
        """Load a trained model and tokenizer"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model found at {path}")
                
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,  # Only look for local files
                torch_dtype=torch.float32
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True  # Only look for local files
            )
            
            # Initialize environment
            self.env = PokerEnvironment()
            
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing new model...")
            self.__init__()  # Reinitialize with default model 