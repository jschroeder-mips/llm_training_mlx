# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx",
#     "mlx-lm",
#     "python-dotenv",
#     "sentencepiece",
# ]
# ///

"""
Fine-tune Microsoft Phi-3-mini to speak like a pirate using MLX and LoRA.

This script trains Phi-3 on pirate speak conversations from the cleaned
GPT007/Pirate_speak dataset.

Usage:
    # First, prepare the dataset
    uv run prepare_pirate_dataset.py
    
    # Then run training
    uv run train_phi3_pirate.py

Requirements:
- Accept Phi-3's terms at https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Set HF_TOKEN in .env file with token that has accepted terms
"""

import json
import subprocess
import sys
import os

from dotenv import load_dotenv
from mlx_lm import generate, load

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
PIRATE_DATA_FILE = "data/pirate_conversations.json"
ADAPTER_FILE = "adapters/phi3_pirate/adapters.npz"
TOKENIZER_CONFIG = {"trust_remote_code": True}

load_dotenv()  # Load optional HF_TOKEN and other overrides from a local .env

print(f"Loading tokenizer for formatting: {MODEL_NAME}")
# We only need the tokenizer for data prep, not the full model yet
# But mlx_lm.load returns both. We'll unload the model to save RAM for training.
model, tokenizer = load(MODEL_NAME, tokenizer_config=TOKENIZER_CONFIG)

# Phi-3-specific: ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✓ Set pad_token to eos_token for Phi-3")

del model # Free up memory

# --- 1. Prepare Dataset for MLX ---
# Load the cleaned pirate conversations dataset
print("Loading pirate conversations dataset...")

if not os.path.exists(PIRATE_DATA_FILE):
    print(f"Error: {PIRATE_DATA_FILE} not found!")
    print("Please run: uv run prepare_pirate_dataset.py")
    sys.exit(1)

with open(PIRATE_DATA_FILE, 'r') as f:
    pirate_data = json.load(f)

train_conversations = pirate_data['train']
val_conversations = pirate_data['valid']

print(f"Train examples: {len(train_conversations)}, Validation examples: {len(val_conversations)}")

# Format for MLX training
print("Formatting for Phi-3...")
train_data = []
for conv in train_conversations:
    messages = [
        {"role": "user", "content": conv['user']},
        {"role": "assistant", "content": conv['assistant']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    train_data.append({"text": text})

val_data = []
for conv in val_conversations:
    messages = [
        {"role": "user", "content": conv['user']},
        {"role": "assistant", "content": conv['assistant']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    val_data.append({"text": text})

# Save to JSONL files in data/ directory
os.makedirs("data", exist_ok=True)

with open("data/train_phi3_pirate.jsonl", "w") as f:
    for line in train_data:
        json.dump(line, f); f.write('\n')

with open("data/valid_phi3_pirate.jsonl", "w") as f:
    for line in val_data:
        json.dump(line, f); f.write('\n')

print(f"✓ Saved formatted data to data/train_phi3_pirate.jsonl and data/valid_phi3_pirate.jsonl")

# --- 2. Training ---
print("\nStarting training on Metal...")
print("Note: Phi-3-mini (3.8B params) training on pirate conversations")

# Create a YAML configuration file for mlx_lm.lora
# With only 89 training examples, we'll train for fewer iterations
os.makedirs("configs", exist_ok=True)
os.makedirs(os.path.dirname(ADAPTER_FILE), exist_ok=True)

config_content = f"""
model: {MODEL_NAME}
train: true
data: data
train_file: train_phi3_pirate.jsonl
valid_file: valid_phi3_pirate.jsonl
batch_size: 2
iters: 500
learning_rate: 1e-5
steps_per_eval: 50
adapter_path: {ADAPTER_FILE}
save_every: 100
lora_parameters:
  rank: 16
  alpha: 16
  dropout: 0.0
  scale: 16.0
"""

with open("configs/lora_config_phi3_pirate.yaml", "w") as f:
    f.write(config_content)

# We use subprocess to call mlx_lm.lora directly.
# This avoids internal API instability and manages memory better.
cmd = [
    sys.executable, "-m", "mlx_lm.lora",
    "--config", "configs/lora_config_phi3_pirate.yaml"
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

print("Training complete. Adapters saved to", ADAPTER_FILE)

# --- 3. Test Inference ---
print("\n--- Testing Inference ---")
# We reload the model with the adapters we just trained
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_FILE, tokenizer_config=TOKENIZER_CONFIG)

test_queries = [
    "Hello, how are you today?",
    "I need to find the treasure on the island.",
    "The weather is beautiful and the sea is calm."
]

for test_query in test_queries:
    print(f"\n🏴‍☠️ Input: {test_query}")
    messages = [{"role": "user", "content": f"Translate the following to pirate speak:\n{test_query}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
    print(f"🦜 Pirate: {response}")
