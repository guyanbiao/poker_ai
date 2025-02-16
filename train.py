import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from poker.training.environment import PokerEnvironment
from poker.models.llm_agent import LLMPokerAgent

def setup_model():
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print(f"Loading model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def train_poker_model(model, tokenizer, num_episodes=100):
    print("Initializing poker environment...")
    env = PokerEnvironment(num_players=6, starting_stack=1000, small_blind=5)
    
    # Replace the default agents with our RL-trained model
    for player_id in env.agents:
        env.agents[player_id] = LLMPokerAgent(
            player_id=player_id,
            personality="learner",
            model=model,
            tokenizer=tokenizer
        )
    
    print("\nStarting training session...")
    reward_history = env.run_training_session(num_episodes)
    
    # Analyze results
    total_rewards = {i: 0 for i in range(env.game.num_players)}
    for episode_rewards in reward_history:
        for player_id, reward in episode_rewards.items():
            total_rewards[player_id] += reward
    
    print("\nTraining Results:")
    for player_id, total_reward in total_rewards.items():
        print(f"Player {player_id} total reward: {total_reward}")
        print(f"Average reward per episode: {total_reward / num_episodes:.2f}")

def main():
    start_time = time.time()
    
    print("Step 1: Setting up model...")
    model, tokenizer = setup_model()
    
    print("\nStep 2: Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print("\nStep 3: Starting poker training...")
    train_poker_model(model, tokenizer)
    
    print("\nStep 4: Saving model...")
    model.save_pretrained("./poker-model-final")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 