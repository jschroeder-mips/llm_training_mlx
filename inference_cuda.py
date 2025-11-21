"""
Interactive inference script for CUDA/PyTorch fine-tuned RISC-V model.

This script loads the fine-tuned LoRA adapters and provides an interactive
interface for generating RISC-V assembly instructions.

Requirements:
    pip install torch transformers peft python-dotenv

Usage:
    python inference_cuda.py
    python inference_cuda.py --adapter_path ./adapters_cuda/final
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER_PATH = "./adapters_cuda/final"

load_dotenv()

def load_model(adapter_path):
    """Load base model and apply LoRA adapters."""
    print(f"Loading base model: {MODEL_NAME}")
    print("This may take a minute on first run...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapters
    print(f"Loading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapters for faster inference
    model = model.merge_and_unload()
    model.eval()
    
    print("✓ Model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, query, max_tokens=150):
    """Generate RISC-V assembly instruction from natural language query."""
    # Format as chat
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode (skip the prompt)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="RISC-V Assembly Code Assistant (CUDA)")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to fine-tuned LoRA adapters"
    )
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Running on CPU (will be slow).")
    else:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_model(args.adapter_path)
    
    print("\n" + "=" * 60)
    print("RISC-V Assembly Code Assistant (CUDA)")
    print("=" * 60)
    print("\nType your query or 'quit' to exit.")
    print("\nNote: Use placeholder register names (rs1, rs2, rd, imm)")
    print("Example: 'Adds the values in rs1 and rs2, stores the result in rd'\n")
    
    while True:
        # Get user input
        user_input = input("\nYour query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Format the query
        query = f"Write the RISC-V assembly instruction for the following operation:\n{user_input}"
        
        # Generate response
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, query)
        
        print(f"\n{'=' * 60}")
        print(response)
        print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
