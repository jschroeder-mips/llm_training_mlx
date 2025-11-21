# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.34.0",
#     "peft>=0.5.0",
#     "datasets>=2.14.0",
#     "bitsandbytes>=0.41.0",
#     "accelerate>=0.23.0",
#     "python-dotenv",
#     "sentencepiece",
#     "protobuf",
# ]
# ///

"""
Fine-tune Mistral-7B-Instruct-v0.3 for RISC-V assembly generation using LoRA on NVIDIA GPUs.

This is the CUDA/PyTorch equivalent of train_mlx.py for Linux/Windows systems with NVIDIA GPUs.
Uses Hugging Face Transformers + PEFT + bitsandbytes for memory-efficient training.

Requirements:
- NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090, A6000, A100, etc.)
- CUDA 11.8+ and compatible PyTorch
- Python 3.10+

Usage:
    uv run train_cuda.py

Expected training time:
- RTX 4090: ~15-20 minutes (600 iterations)
- RTX 3090: ~20-30 minutes
- T4 (Colab): ~45-60 minutes
"""

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME = "davidpirkl/riscv-instruction-specification"
OUTPUT_DIR = "./adapters_cuda"
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"

# Load environment variables (HF_TOKEN)
load_dotenv()

print("=" * 60)
print("CUDA/PyTorch LoRA Fine-Tuning for RISC-V Assembly")
print("=" * 60)

# Check CUDA availability
if not torch.cuda.is_available():
    print("\n⚠️  WARNING: CUDA not available. This will be VERY slow on CPU.")
    print("Make sure you have:")
    print("  1. NVIDIA GPU installed")
    print("  2. CUDA drivers installed")
    print("  3. PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(1)
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n✓ GPU detected: {gpu_name}")
    print(f"✓ VRAM available: {gpu_memory:.1f} GB")
    
    if gpu_memory < 16:
        print("\n⚠️  WARNING: Less than 16GB VRAM detected.")
        print("Training may fail with OOM. Consider:")
        print("  - Using batch_size=1")
        print("  - Using gradient_accumulation_steps=4")
        print("  - Reducing max_seq_length")

# --- 1. Prepare Dataset ---
print("\n" + "=" * 60)
print("STEP 1: Preparing Dataset")
print("=" * 60)

print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)

# Mistral doesn't have a pad token by default, use eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"✓ Loaded {len(dataset)} examples")

# Format dataset
print("\nFormatting dataset with chat template...")
data_list = []
for item in dataset:
    user_content = f"Write the RISC-V assembly instruction for the following operation:\n{item['description']}"
    assistant_content = item['instructions']
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # Format with chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    data_list.append({"text": text})

# Split into train/validation (90/10)
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
valid_data = data_list[split_idx:]

print(f"✓ Split: {len(train_data)} training, {len(valid_data)} validation")

# Save to JSONL files
with open(TRAIN_FILE, "w") as f:
    for line in train_data:
        json.dump(line, f)
        f.write('\n')

with open(VALID_FILE, "w") as f:
    for line in valid_data:
        json.dump(line, f)
        f.write('\n')

print(f"✓ Saved {TRAIN_FILE} and {VALID_FILE}")

# --- 2. Load Model with Quantization ---
print("\n" + "=" * 60)
print("STEP 2: Loading Model with 4-bit Quantization")
print("=" * 60)

# Configure 4-bit quantization (QLoRA)
# This reduces memory usage significantly while maintaining quality
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",              # Normal Float 4 (better than int4)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation dtype
    bnb_4bit_use_double_quant=True,         # Nested quantization for more memory savings
)

print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")
print("This may take 1-2 minutes on first run (downloading ~14GB model)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",                      # Automatically use available GPUs
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print("✓ Model loaded successfully")

# Prepare model for k-bit training (required for QLoRA)
model = prepare_model_for_kbit_training(model)
print("✓ Model prepared for 4-bit training")

# --- 3. Configure LoRA ---
print("\n" + "=" * 60)
print("STEP 3: Configuring LoRA Adapters")
print("=" * 60)

# LoRA configuration
# These parameters match the MLX version for consistency
lora_config = LoraConfig(
    r=16,                           # LoRA rank (dimensionality of adapters)
    lora_alpha=16,                  # Scaling factor
    target_modules=[                # Which layers to apply LoRA to
        "q_proj",                   # Query projection (attention)
        "k_proj",                   # Key projection (attention)
        "v_proj",                   # Value projection (attention)
        "o_proj",                   # Output projection (attention)
        "gate_proj",                # Gate projection (MLP)
        "up_proj",                  # Up projection (MLP)
        "down_proj",                # Down projection (MLP)
    ],
    lora_dropout=0.0,               # No dropout (same as MLX version)
    bias="none",                    # Don't adapt bias parameters
    task_type="CAUSAL_LM",          # Causal language modeling task
)

print("LoRA Configuration:")
print(f"  Rank: {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / total_params

print(f"\n✓ LoRA adapters applied")
print(f"  Trainable parameters: {trainable_params:,} ({trainable_percent:.3f}%)")
print(f"  Total parameters: {total_params:,}")

# --- 4. Prepare Training Data ---
print("\n" + "=" * 60)
print("STEP 4: Tokenizing Dataset")
print("=" * 60)

# Load formatted datasets
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
valid_dataset = load_dataset("json", data_files=VALID_FILE, split="train")

# Tokenization function
def tokenize_function(examples):
    """Tokenize text and prepare for causal language modeling."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,             # Maximum sequence length
        padding=False,              # Don't pad here, data collator will handle it
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing training data...")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train",
)

print("Tokenizing validation data...")
valid_dataset = valid_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing valid",
)

print(f"✓ Tokenization complete")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                      # Not masked language modeling
)

# --- 5. Configure Training ---
print("\n" + "=" * 60)
print("STEP 5: Configuring Training Parameters")
print("=" * 60)

# Training arguments
# These are tuned to match the MLX training results
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=2,             # ~2 epochs with our dataset size
    max_steps=600,                  # Same as MLX version
    
    # Batch sizes
    per_device_train_batch_size=1,  # Batch size per GPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch_size = 1 * 2 = 2 (matches MLX)
    
    # Optimization
    learning_rate=1e-5,             # Same as MLX version
    lr_scheduler_type="constant",   # Constant learning rate
    warmup_steps=0,                 # No warmup
    
    # Precision
    bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if available
    fp16=not torch.cuda.is_bf16_supported(),  # Otherwise fp16
    
    # Logging
    logging_steps=50,               # Log every 50 steps (same as steps_per_eval)
    logging_first_step=True,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,                  # Evaluate every 50 steps
    
    # Checkpointing
    save_strategy="steps",
    save_steps=100,                 # Save every 100 steps
    save_total_limit=6,             # Keep last 6 checkpoints
    
    # Other
    load_best_model_at_end=False,
    report_to="none",               # Disable wandb/tensorboard
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,   # Disable to match MLX behavior
)

print("Training Configuration:")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Precision: {'bfloat16' if training_args.bf16 else 'float16'}")
print(f"  Save every: {training_args.save_steps} steps")
print(f"  Eval every: {training_args.eval_steps} steps")

# --- 6. Train ---
print("\n" + "=" * 60)
print("STEP 6: Training")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

print("\nStarting training...")
print("This will take ~15-30 minutes depending on your GPU.")
print("Progress will be logged every 50 steps.\n")

# Train the model
trainer.train()

print("\n✓ Training complete!")

# --- 7. Save Final Model ---
print("\n" + "=" * 60)
print("STEP 7: Saving Adapters")
print("=" * 60)

# Save LoRA adapters
final_adapter_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)

print(f"✓ Adapters saved to: {final_adapter_dir}")
print(f"  - adapter_model.safetensors (~80MB)")
print(f"  - adapter_config.json")
print(f"  - tokenizer files")

# --- 8. Test Inference ---
print("\n" + "=" * 60)
print("STEP 8: Testing Inference")
print("=" * 60)

print("\nLoading model for inference...")

# Merge adapters for faster inference
model = model.merge_and_unload()
model.eval()

# Test generation
test_query = "Write the RISC-V assembly instruction for the following operation:\nAdds the values in rs1 and rs2, stores the result in rd"
messages = [{"role": "user", "content": test_query}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"\nTest Query: {test_query}")
print("\nGenerating response...")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,          # Greedy decoding for consistency
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n" + "=" * 60)
print(response.strip())
print("=" * 60)

print("\n✅ Training pipeline complete!")
print(f"\nYour fine-tuned adapters are in: {final_adapter_dir}")
print("\nTo use them:")
print(f"  python inference_cuda.py --adapter_path {final_adapter_dir}")
