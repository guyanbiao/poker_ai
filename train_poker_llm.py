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
        
        # Load tokenizer and model with PyTorch backend
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
        
        self.env = PokerEnvironment()
        
    def format_state(self, state: Dict) -> str:
        """Convert poker state to text format for LLM"""
        return f"""
        Your hand: {' '.join(state['hand'])}
        Pot: ${state['pot']}
        Current bet: ${state['current_bet']}
        What action would you like to take? (fold/call/raise)
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
            
            if episode % 10 == 0:  # Print more frequently
                print(f"Episode {episode}, Reward: {episode_reward}, Loss: {episode_loss:.4f}")
        
        # Save the trained model
        self.save_model()

    def _parse_action(self, action_text: str) -> str:
        """Parse model output to valid poker action"""
        valid_actions = ['fold', 'call', 'raise']
        for valid_action in valid_actions:
            if valid_action in action_text.lower():
                return valid_action
        return 'fold'  # Default action if no valid action found

    def save_model(self, path: str = "poker_model"):
        """Save the trained model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "poker_model"):
        """Load a trained model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}") 