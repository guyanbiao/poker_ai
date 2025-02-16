import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def setup_model():
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load DeepSeek Coder model
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

def prepare_dataset(tokenizer, data_path="training_data.json"):
    print(f"Loading dataset from: {data_path}")
    dataset = load_dataset("json", data_files=data_path)
    
    def preprocess_function(examples):
        # Format examples with DeepSeek's instruction format
        texts = [
            f"### Instruction: {instruction}\n\n### Response: {response}"
            for instruction, response in zip(examples["instruction"], examples["response"])
        ]
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    print("Processing dataset...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Processing dataset"
    )
    
    return processed_dataset

def setup_lora():
    # LoRA configuration for efficient fine-tuning
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def main():
    start_time = time.time()
    
    print("Step 1: Setting up model...")
    model, tokenizer = setup_model()
    
    print("\nStep 2: Applying LoRA...")
    lora_config = setup_lora()
    model = get_peft_model(model, lora_config)
    
    print("\nStep 3: Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    print("\nStep 4: Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./deepseek-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        max_steps=1000,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        logging_dir="./logs",
        save_total_limit=2,
    )
    
    print("\nStep 5: Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("\nStep 6: Starting training...")
    trainer.train()
    
    print("\nStep 7: Saving model...")
    trainer.save_model("./deepseek-finetuned-final")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 