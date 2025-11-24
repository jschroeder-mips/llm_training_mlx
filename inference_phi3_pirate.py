#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mlx",
#   "mlx-lm",
#   "python-dotenv",
# ]
# ///

"""
Interactive inference script for Phi-3 trained on pirate speak.

Usage:
    uv run inference_phi3_pirate.py

The script will load the pirate-trained Phi-3 adapter and translate
your input into pirate speak. Type 'quit' to exit.
"""

import os
from mlx_lm import load, generate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_FILE = "adapters/phi3_pirate/adapters.npz"

def main():
    print("🏴‍☠️ Loading Phi-3 Pirate Speak Model...")
    print(f"Model: {MODEL_NAME}")
    print(f"Adapter: {ADAPTER_FILE}\n")
    
    # Check if adapter exists
    if not os.path.exists(ADAPTER_FILE):
        print(f"❌ Error: Adapter not found at {ADAPTER_FILE}")
        print("Please run train_phi3_pirate.py first to create the adapter.")
        return
    
    # Load model with pirate adapter
    model, tokenizer = load(
        MODEL_NAME,
        adapter_path=ADAPTER_FILE,
        tokenizer_config={"trust_remote_code": True}
    )
    
    print("✓ Model loaded successfully!\n")
    print("=" * 60)
    print("🏴‍☠️ PIRATE SPEAK TRANSLATOR 🏴‍☠️")
    print("=" * 60)
    print("Enter text to translate into pirate speak.")
    print("Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n⚓ Fair winds and following seas, matey!")
            break
        
        if not user_input:
            continue
        
        # Format prompt with chat template
        messages = [
            {"role": "user", "content": f"Translate the following to pirate speak:\n{user_input}"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate pirate response
        print("🦜 Pirate: ", end="", flush=True)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=False
        )
        # Strip end tokens from response
        response = response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
        print(response)
        print()

if __name__ == "__main__":
    main()
