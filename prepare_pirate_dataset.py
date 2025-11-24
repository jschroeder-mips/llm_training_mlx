# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "python-dotenv",
# ]
# ///

"""
Download and reformat the GPT007/Pirate_speak dataset for training with Phi-3 and Mistral.

The original dataset contains Llama 3 formatted conversations. This script extracts
the actual pirate dialogue and creates clean training examples.

Usage:
    uv run prepare_pirate_dataset.py
"""

import json
import re
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

print("Loading GPT007/Pirate_speak dataset...")
dataset = load_dataset("GPT007/Pirate_speak", split="train")

print(f"Dataset size: {len(dataset)} examples")
print(f"Dataset columns: {dataset.column_names}")

# The dataset has conversations in Llama 3 format
# We need to extract the user/assistant pairs and clean them up
data_list = []

for idx, item in enumerate(dataset):
    # Get the text field (should be the first/only column)
    raw_text = item[dataset.column_names[0]]
    
    # Parse Llama 3 format: <|start_header_id|>user<|end_header_id|> ... <|eot_id|>
    # Extract user and assistant messages
    user_pattern = r'<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>'
    assistant_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
    
    user_matches = re.findall(user_pattern, raw_text, re.DOTALL)
    assistant_matches = re.findall(assistant_pattern, raw_text, re.DOTALL)
    
    # Take the first user/assistant pair if available
    if user_matches and assistant_matches:
        user_text = user_matches[0].strip()
        assistant_text = assistant_matches[0].strip()
        
        # Skip if either is too short or empty
        if len(user_text) > 10 and len(assistant_text) > 10:
            data_list.append({
                "user": user_text,
                "assistant": assistant_text
            })
            
            if idx < 3:  # Print first 3 examples
                print(f"\n--- Example {idx + 1} ---")
                print(f"User: {user_text[:100]}...")
                print(f"Assistant: {assistant_text[:100]}...")

print(f"\nExtracted {len(data_list)} valid conversation pairs")

# Split into train/validation (90/10)
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")

# Save to data directory
import os
os.makedirs("data", exist_ok=True)

# Save as simple JSON (not JSONL yet - that's for the training scripts)
with open("data/pirate_conversations.json", "w") as f:
    json.dump({
        "train": train_data,
        "valid": val_data
    }, f, indent=2)

print("\n✓ Saved to data/pirate_conversations.json")
print("\nDataset structure:")
print("  - Each example has 'user' and 'assistant' fields")
print("  - Training scripts will format these with model-specific chat templates")
print("  - User messages: Questions/prompts in various styles")
print("  - Assistant messages: Responses in pirate speak")
