# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx",
#     "mlx-lm",
#     "datasets",
# ]
# ///

import json

from datasets import load_dataset
from mlx_lm import generate, load, train
from mlx_lm.tuner.utils import linear_to_lora_layers

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME = "davidpirkl/riscv-instruction-specification"
ADAPTER_FILE = "adapters.npz"
TOKENIZER_CONFIG = {"trust_remote_code": True}

print(f"Loading model: {MODEL_NAME}")
# MLX handles 4-bit quantization natively with tokenizer_config
model, tokenizer = load(MODEL_NAME, tokenizer_config=TOKENIZER_CONFIG)

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

# --- 2. Freeze Model & Convert to LoRA ---
# Freezes the main model and adds LoRA adapters to linear layers
model.freeze()
linear_to_lora_layers(model, list(range(len(model.layers))), {"q_proj", "v_proj"}, r=16, alpha=16)

# --- 3. Training ---
print("Starting training on Metal (M4)...")

# MLX's train function is very efficient
train(
    model=model,
    tokenizer=tokenizer,
    optimizer=None, # Default optimizer
    train_dataset="train.jsonl",
    valid_dataset="valid.jsonl",
    max_seq_length=512,
    batch_size=4,       # Adjust based on your RAM (4 is usually safe for 7B on 16GB RAM)
    iters=600,          # 600 iterations is roughly 1 epoch for 2k rows with batch size 4
    learning_rate=1e-5,
    steps_per_eval=50,
    adapter_file=ADAPTER_FILE, # Saves the LoRA weights here
)

print("Training complete. Adapters saved to", ADAPTER_FILE)

# --- 4. Test Inference ---
print("\n--- Testing Inference ---")
# We reload the model with the adapters we just trained
model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_FILE, tokenizer_config=TOKENIZER_CONFIG)

test_query = "Write the RISC-V assembly instruction for the following operation:\nAdd the values in register s1 and s2 and store them in t0"
messages = [{"role": "user", "content": test_query}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=True)
print(response)