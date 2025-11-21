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

import json
import subprocess
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from mlx_lm import generate, load

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME = "davidpirkl/riscv-instruction-specification"
ADAPTER_FILE = "adapters.npz"
TOKENIZER_CONFIG = {"trust_remote_code": True}

load_dotenv()  # Load optional HF_TOKEN and other overrides from a local .env

print(f"Loading tokenizer for formatting: {MODEL_NAME}")
# We only need the tokenizer for data prep, not the full model yet
# But mlx_lm.load returns both. We'll unload the model to save RAM for training.
model, tokenizer = load(MODEL_NAME, tokenizer_config=TOKENIZER_CONFIG)
del model # Free up memory

# --- 1. Prepare Dataset for MLX ---
# MLX expects a specific format (ChatML or simple text). 
# We will format it to the chat template and save a local JSONL file.
print("Formatting dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

data_list = []
for item in dataset:
    # We construct the full prompt text that the model should learn
    # This mimics the "apply_chat_template" logic
    user_content = f"Write the RISC-V assembly instruction for the following operation:\n{item['description']}"
    assistant_content = item['instructions']
    
    # Simple ChatML-like format manual construction if tokenizer.apply_chat_template isn't perfect
    # But usually, for text completion training in MLX, we just want text.
    # We will use the tokenizer to ensure it matches the model's expected format.
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

# Save to JSONL files
with open("train.jsonl", "w") as f:
    for line in train_data:
        json.dump(line, f); f.write('\n')

with open("valid.jsonl", "w") as f:
    for line in val_data:
        json.dump(line, f); f.write('\n')

# --- 2. Training ---
print("Starting training on Metal (M4)...")

# Create a YAML configuration file for mlx_lm.lora
# This allows us to specify advanced LoRA settings like target modules.
config_content = f"""
model: {MODEL_NAME}
train: true
data: .
batch_size: 4
iters: 600
learning_rate: 1e-5
steps_per_eval: 50
adapter_path: {ADAPTER_FILE}
save_every: 100
num_layers: -1  # -1 means all layers
lora_parameters:
  rank: 16
  alpha: 16
  dropout: 0.0
  keys: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
"""

with open("lora_config.yaml", "w") as f:
    f.write(config_content)

# We use subprocess to call mlx_lm.lora directly.
# This avoids internal API instability and manages memory better.
cmd = [
    sys.executable, "-m", "mlx_lm.lora",
    "--config", "lora_config.yaml"
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