# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx",
#     "mlx-lm",
#     "python-dotenv",
#     "sentencepiece",
# ]
# ///

from mlx_lm import generate, load
from dotenv import load_dotenv

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_FILE = "adapters/mistral/adapters.npz"
TOKENIZER_CONFIG = {"trust_remote_code": True}

load_dotenv()

print("Loading fine-tuned model...")
model, tokenizer = load(
    MODEL_NAME, 
    adapter_path=ADAPTER_FILE, 
    tokenizer_config=TOKENIZER_CONFIG
)

print("\n" + "="*60)
print("RISC-V Assembly Code Assistant")
print("="*60)
print("\nType your query or 'quit' to exit.\n")

while True:
    # Get user input
    user_input = input("\nYour query: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    # Format the prompt
    query = f"Write the RISC-V assembly instruction for the following operation:\n{user_input}"
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Generate response
    print("\nGenerating response...")
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=150,
        verbose=False
    )
    
    print(f"\n{'='*60}")
    print(response)
    print(f"{'='*60}")
