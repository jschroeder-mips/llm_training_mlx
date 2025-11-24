# llm_training_mlx

A practical guide to fine-tuning large language models (LLMs) on Apple Silicon. This project demonstrates how to specialize `mistralai/Mistral-7B-Instruct-v0.3` (a general-purpose 7 billion parameter model) into a RISC-V assembly code assistant using parameter-efficient fine-tuning.

**📱 Platform Support:**
- **Apple Silicon** (M1/M2/M3/M4): Use `train_mlx.py` with MLX framework → [See instructions below](#setup)
- **NVIDIA GPUs** (Windows/Linux): Use `train_cuda.py` with PyTorch → [See CUDA_SETUP.md](CUDA_SETUP.md)
- **Google Colab** (Free/Pro): Use `train_colab.ipynb` notebook → [Open in Colab](https://colab.research.google.com/github/jschroeder-mips/llm_training_mlx/blob/main/train_colab.ipynb)

## What is Fine-Tuning?

**Fine-tuning** adapts a pre-trained model to perform specialized tasks by training it on domain-specific data. Instead of training a model from scratch (which requires massive compute and data), you start with a capable foundation model and teach it new skills.

**Why fine-tune instead of prompting?**
- More reliable outputs for specialized domains
- Can learn patterns not well-represented in the base model's training data
- Reduces need for complex prompts or examples in every query
- Model "internalizes" domain knowledge

**This project uses:**
- **Base model**: Mistral-7B-Instruct-v0.3 (general-purpose conversational AI)
- **Target domain**: RISC-V assembly language instruction generation  
- **Method**: LoRA (Low-Rank Adaptation) - a parameter-efficient technique
- **Hardware**: Apple Silicon (M1/M2/M3/M4) using Metal GPU acceleration via [MLX](https://github.com/ml-explore/mlx)
- **Tooling**: [uv](https://github.com/astral-sh/uv) for dependency management

## Understanding LoRA (Low-Rank Adaptation)

Training billions of parameters is expensive. **LoRA** is a clever trick that makes fine-tuning practical:

**Traditional Fine-Tuning:**
- Modifies all 7.2 billion parameters in the model
- Requires enormous memory and compute
- Results in a full-size model copy for each task

**LoRA Approach:**
- Freezes the original 7.2B parameters (they stay unchanged)
- Adds small "adapter" matrices (20.97M parameters - just 0.29% of the model!)
- Only trains these tiny adapters
- At inference time, the adapters modify the base model's behavior on-the-fly

**Analogy**: Instead of rewriting an entire encyclopedia for a specialized topic, you're adding sticky notes with corrections and additions. The original stays intact, but readers see the modified version.

**What you'll find in `adapters/mistral/adapters.npz/`:**
- `adapters.safetensors` (80MB) - The trained LoRA weights
- `adapter_config.json` - Metadata about the LoRA configuration (rank, alpha, target layers)

The base Mistral model (~14GB) downloads separately from Hugging Face and is cached locally. Your specialized model = base model + adapters.

## How This Project Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION (train_mlx.py)                         │
│    Download RISC-V dataset → Format with chat template     │
│    → Split into data/train.jsonl + data/valid.jsonl        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MODEL LOADING                                            │
│    Base: Mistral-7B-Instruct-v0.3 (7.2B params)            │
│    Freeze all parameters (no base model modification)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. LoRA INJECTION                                           │
│    Add adapter matrices to linear layers:                  │
│    - q_proj, k_proj, v_proj (attention query/key/value)   │
│    - o_proj (attention output)                             │
│    - gate_proj, up_proj, down_proj (feed-forward network) │
│                                                             │
│    Config: rank=16, alpha=16, scale=16.0                   │
│    Trainable params: 20.97M (0.29%)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. TRAINING (MLX on Metal GPU)                             │
│    600 iterations, batch_size=2, learning_rate=1e-5        │
│    Save checkpoints every 100 iterations                    │
│    Final loss: ~0.94                                        │
│    Peak memory: ~17.6GB (M4 MacBook)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. INFERENCE (inference.py)                                 │
│    Load base model + adapters → Generate RISC-V assembly   │
└─────────────────────────────────────────────────────────────┘
```

### Key Files Explained

**`train_mlx.py`** - The complete training pipeline:
1. Downloads `davidpirkl/riscv-instruction-specification` dataset
2. Formats each example as a conversation (user describes operation → assistant provides instruction)
3. Generates `configs/lora_config.yaml` with training parameters
4. Launches MLX's LoRA training via subprocess
5. Tests the trained model with a sample query

**`inference.py`** - Interactive testing script:
- Loads the base model + your trained adapters
- Accepts natural language queries
- Generates RISC-V assembly instructions
- Uses the same chat template format as training

**Training Data (`data/train.jsonl` / `data/valid.jsonl`)**:
Each line is a JSON object with a `text` field containing a formatted conversation:
```
<s>[INST] Write the RISC-V assembly instruction for the following operation:
Adds the values in rs1 and rs2, stores the result in rd.[/INST] add rd, rs1, rs2</s>
```

**Adapters (`adapters/mistral/adapters.npz/`)**:
- `adapters.safetensors` - The learned LoRA weights (80MB)
- `adapter_config.json` - Configuration metadata

**`configs/lora_config.yaml`** (auto-generated):
Training hyperparameters including model path, batch size, learning rate, LoRA rank/alpha/scale, and target modules.

## Prerequisites

**Hardware:**
- Apple Silicon Mac (M1/M2/M3/M4) with 16GB+ unified memory recommended
- Training uses Metal GPU acceleration via MLX framework
- Expect ~17-18GB peak memory usage with batch_size=2

**Software:**
- Python 3.11+
- A Hugging Face account with access token
- `uv` package manager (or use pip directly)

## Setup

### 1. Install uv (Recommended)

`uv` is a fast Python package manager that handles dependencies via inline metadata (PEP 722) in the scripts.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `~/.local/bin` to your `PATH` if prompted, then verify:
```bash
uv --version
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen PowerShell, then verify:
```powershell
uv --version
```

**Alternative: Use pip directly**
```bash
pip install mlx mlx-lm datasets python-dotenv sentencepiece
```

### 2. Authenticate with Hugging Face

Models and datasets download from Hugging Face, which requires authentication. Get your token from: https://huggingface.co/settings/tokens (requires "Read" scope)

**Option A: Login via CLI**
```bash
uv run huggingface-cli login
# Or if using pip: huggingface-cli login
```

**Option B: Environment Variable**
```bash
# macOS/Linux
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

**Option C: .env File** (recommended for persistence)

Create a `.env` file in the repo root:
```
HF_TOKEN=hf_your_token_here
```

### 3. Clone and Initialize

```bash
git clone https://github.com/jschroeder-mips/llm_training_mlx.git
cd llm_training_mlx

# If using uv (one-time setup)
uv sync
```

## Training Workflow

Run the training script through uv so dependencies are isolated and reproducible:

```bash
uv run train_mlx.py
```

**What happens during training:**

1. **Download Dataset** (~5 seconds)
   - Fetches RISC-V instruction specifications from Hugging Face
   - 590 examples describing assembly operations

2. **Format Data** (~10 seconds)
   - Converts each example to chat format using Mistral's template
   - Creates `data/train.jsonl` (531 examples) and `data/valid.jsonl` (59 examples)
   - Each line: `{"text": "<s>[INST] prompt [/INST] response</s>"}`

3. **Load Base Model** (~30 seconds, first run only)
   - Downloads Mistral-7B-Instruct-v0.3 (~14GB) to `~/.cache/huggingface`
   - Subsequent runs use cached model

4. **Apply LoRA** (~5 seconds)
   - Freezes all 7.2B base parameters
   - Injects adapter matrices into 7 linear layer types
   - Creates 20.97M trainable parameters (0.29% of model)

5. **Train** (~20-30 minutes on M4)
   - 600 iterations with batch_size=2
   - Evaluates on validation set every 50 steps
   - Saves checkpoints every 100 steps
   - Final training loss: ~0.94

6. **Test Inference** (~5 seconds)
   - Loads trained adapters
   - Generates a sample RISC-V instruction
   - Verifies model works correctly

**Training Output Example:**
```
Iter 100: Train loss 1.523, Learning Rate 1.000e-05, It/sec 0.621, Peak mem 17.2 GB
Iter 100: Saved adapter weights to adapters/mistral/adapters.npz/adapters.safetensors
...
Iter 600: Train loss 0.944, Learning Rate 1.000e-05, It/sec 0.643, Peak mem 17.6 GB
Training complete. Adapters saved to adapters/mistral/adapters.npz
```

**Artifacts created:**
- `data/train.jsonl` / `data/valid.jsonl` - Formatted training data (490KB + 57KB)
- `configs/lora_config.yaml` - Training configuration
- `adapters/mistral/adapters.npz/adapters.safetensors` - Trained LoRA weights (80MB)
- `adapters/mistral/adapters.npz/adapter_config.json` - Adapter metadata

## Inference

Test the fine-tuned model interactively:

```bash
uv run inference.py
```

The model expects queries matching the training data format. Use **placeholder register names** (rs1, rs2, rd, imm) rather than specific ones (t0, s1, etc.):

**Example Session:**
```
Your query: Adds the values in rs1 and rs2, stores the result in rd

Generating response...
============================================================
add rd, rs1, rs2
============================================================

Your query: Subtracts the value in rs2 from rs1, stores the result in rd

Generating response...
============================================================
sub rd, rs1, rs2
============================================================
```

**More Example Queries:**
- "Loads a word from memory at address rs1 into rd" → `lw rd, 0(rs1)`
- "Multiplies the values in two registers (rs1, rs2) and stores the result in rd" → `mul rd, rs1, rs2`
- "Branches to label if rs1 equals rs2" → `beq rs1, rs2, label`

Type `quit` or `exit` to stop the inference session.

## Understanding Training Parameters

**Batch Size (batch_size=2):**
- Number of examples processed simultaneously
- Larger = faster training but more memory
- batch_size=2 fits M4 with 16GB memory; reduce to 1 if OOM occurs

**Learning Rate (learning_rate=1e-5):**
- How much to adjust weights each step
- Too high = unstable training, too low = slow convergence
- 1e-5 (0.00001) is conservative and safe for fine-tuning

**Iterations (iters=600):**
- Number of training steps
- With batch_size=2 and 531 training examples, this is ~2 epochs
- More iterations = better learning but risk overfitting

**LoRA Rank (rank=16):**
- Dimensionality of adapter matrices
- Higher rank = more expressive but more parameters and memory
- rank=16 is a sweet spot for 7B models

**LoRA Alpha (alpha=16):**
- Scaling factor for adapter outputs
- Typically set equal to rank
- Controls how much adapters influence the base model

**LoRA Scale (scale=16.0):**
- Additional scaling applied during training
- Required by MLX's implementation
- Usually matches alpha value

## Troubleshooting

**Out of Memory (OOM) Errors:**
```
[METAL] Command buffer execution failed: Insufficient Memory
```
- Reduce `batch_size` in `train_mlx.py` (try 1 instead of 2)
- Close other applications to free memory
- If still failing, reduce LoRA `rank` from 16 to 8

**Hugging Face Authentication Errors:**
```
HTTPError: 401 Client Error: Unauthorized
```
- Run `uv run huggingface-cli login` and enter your token
- Or set `HF_TOKEN` environment variable
- Verify token has "Read" permissions at https://huggingface.co/settings/tokens

**Training Loss Not Decreasing:**
- Check if `Trainable parameters: 0.000%` - indicates LoRA not applied correctly
- Ensure `lora_config.yaml` includes `scale` parameter
- Verify training data format matches expected chat template

**Inference Generates Incomplete Output:**
- Model was trained on placeholder names (rs1, rs2, rd) not specific ones (t0, s1, a0)
- Use queries like "Adds the values in rs1 and rs2" not "Add s1 and s2"
- Check that adapters loaded correctly (no errors during model loading)

**Corrupted Downloads:**
```
File appears to be corrupted
```
- Clear Hugging Face cache: `rm -rf ~/.cache/huggingface`
- Re-run training to download fresh copies

## Customizing for Your Domain

Want to fine-tune for a different task? Here's how to adapt this project:

### 1. Choose Your Dataset

Replace the dataset in `train_mlx.py`:
```python
DATASET_NAME = "your-username/your-dataset"
```

Your dataset needs:
- A "prompt" or "input" field (user query)
- A "response" or "target" field (desired output)

### 2. Update Data Formatting

Modify the formatting loop to match your dataset structure:
```python
for item in dataset:
    user_content = item['your_prompt_field']
    assistant_content = item['your_response_field']
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    data_list.append({"text": text})
```

### 3. Adjust Training Parameters

For different dataset sizes:
- Small dataset (< 500 examples): Increase `iters` to 1000+
- Large dataset (> 5000 examples): May need fewer iterations (300-500)
- Longer outputs: Reduce `batch_size` to avoid OOM

### 4. Test Thoroughly

After training, test with diverse inputs to verify the model learned correctly. Check for:
- Overfitting (perfect training loss but poor on new examples)
- Underfitting (high training loss, poor outputs)
- Format issues (model generates partial or malformed outputs)

## Next Steps

**Experiment with hyperparameters:**
- Try different LoRA ranks (8, 16, 32) to balance memory vs. quality
- Adjust learning rate (1e-4 for faster convergence, 1e-6 for more stability)
- Train longer (1000+ iterations) for better performance

**Merge adapters into base model:**
```python
# Coming soon - instructions for merging LoRA weights
# This creates a standalone model without needing adapter files
```

**Deploy your model:**
- Use `mlx_lm.generate()` in a web API (Flask, FastAPI)
- Integrate into applications that need domain-specific generation
- Share adapters on Hugging Face for others to use

**Try different base models:**
- Replace `MODEL_NAME` with other Mistral variants
- Experiment with Llama, Qwen, or other MLX-compatible models
- Larger models (13B, 70B) require more memory but may perform better

## Resources

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX-LM Guide**: https://github.com/ml-explore/mlx-examples/tree/main/llms
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **CUDA/PyTorch Setup**: [CUDA_SETUP.md](CUDA_SETUP.md) - For Windows/Linux with NVIDIA GPUs
- **Google Colab Notebook**: [train_colab.ipynb](train_colab.ipynb) - Run in browser with free GPU
- **This Repo**: https://github.com/jschroeder-mips/llm_training_mlx

## Platform Comparison

| Platform | Script | Hardware | Training Time | Setup Complexity |
|----------|--------|----------|---------------|------------------|
| **Apple Silicon** | `train_mlx.py` | M1/M2/M3/M4 Mac | ~25-35 min | ⭐ Easy (uv run) |
| **NVIDIA GPU** | `train_cuda.py` | RTX 3090/4090, A100 | ~15-30 min | ⭐⭐ Medium (CUDA setup) |
| **Google Colab** | `train_colab.ipynb` | T4/V100/A100 (cloud) | ~20-60 min | ⭐ Easy (browser) |

All platforms produce equivalent results. Choose based on your available hardware.

## License

See [LICENSE](LICENSE) file for details.
