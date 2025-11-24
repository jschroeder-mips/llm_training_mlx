# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx",
#     "mlx-lm",
#     "datasets",
#     "python-dotenv",
#     "sentencepiece",
# ]
# ///

"""
Fine-tune Microsoft Phi-3-mini on RISC-V assembly using MLX and LoRA.

Phi-3-mini is a 3.8B parameter model that's highly efficient and works well
with MLX's LoRA implementation. It should train faster than Mistral-7B while
maintaining strong performance.

Usage:
    uv run train_phi3.py

Requirements:
- Accept Phi-3's terms at https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Set HF_TOKEN in .env file with token that has accepted terms
"""

import json
import subprocess
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from mlx_lm import generate, load

# --- Configuration ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_NAME = "davidpirkl/riscv-instruction-specification"
ADAPTER_FILE = "adapters/phi3/adapters.npz"
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
# MLX expects a specific format (ChatML or simple text). 
# We will format it to the chat template and save a local JSONL file.
print("Formatting dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

data_list = []
for item in dataset:
    # We construct the full prompt text that the model should learn
    user_content = f"Write the RISC-V assembly instruction for the following operation:\n{item['description']}"
    assistant_content = item['instructions']
    
    # Use the tokenizer's chat template to ensure proper formatting for Phi-3
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    data_list.append({"text": text})

# Split into train/val (MLX likes having a validation set)
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# Save to JSONL files in data/ directory
import os
os.makedirs("data", exist_ok=True)

with open("data/train_phi3.jsonl", "w") as f:
    for line in train_data:
        json.dump(line, f); f.write('\n')

with open("data/valid_phi3.jsonl", "w") as f:
    for line in val_data:
        json.dump(line, f); f.write('\n')

# --- 2. Training ---
print("Starting training on Metal (M4)...")
print("Note: Phi-3-mini (3.8B params) is smaller than Mistral-7B")
print("      Training should be faster with similar or better results")

# Create a YAML configuration file for mlx_lm.lora
# Phi-3 is efficient, so we can potentially use batch_size=4 if memory allows
# Starting conservative with batch_size=2
os.makedirs("configs", exist_ok=True)
os.makedirs(os.path.dirname(ADAPTER_FILE), exist_ok=True)

config_content = f"""
model: {MODEL_NAME}
train: true
data: data
train_file: train_phi3.jsonl
valid_file: valid_phi3.jsonl
batch_size: 2
iters: 600
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

with open("configs/lora_config_phi3.yaml", "w") as f:
    f.write(config_content)

# We use subprocess to call mlx_lm.lora directly.
# This avoids internal API instability and manages memory better.
cmd = [
    sys.executable, "-m", "mlx_lm.lora",
    "--config", "configs/lora_config_phi3.yaml"
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

print("Training complete. Adapters saved to", ADAPTER_FILE)

# --- 3. Test Inference ---
print("\n--- Testing Inference ---")
# We reload the model with the adapters we just trained
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_FILE, tokenizer_config=TOKENIZER_CONFIG)

test_query = "Write the RISC-V assembly instruction for the following operation:\nAdd the values in register s1 and s2 and store them in t0"
messages = [{"role": "user", "content": test_query}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=True)
print(response)
