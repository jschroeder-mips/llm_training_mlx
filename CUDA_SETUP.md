# CUDA/PyTorch Setup Guide

This guide is for Windows and Linux users with NVIDIA GPUs who want to fine-tune the model using CUDA instead of Apple's MLX framework.

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM
  - RTX 4060 Ti 16GB (tight, batch_size=1)
  - RTX 3090 (24GB) ✅ Recommended for learning
  - RTX 4090 (24GB) ✅ Recommended for learning
- **RAM**: 16GB system RAM
- **Storage**: 30GB free space (model cache + checkpoints)

### Recommended for Production
- **GPU**: A6000 (48GB), A100 (40GB/80GB), H100
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ SSD

## Software Requirements

### 1. NVIDIA Drivers
Install the latest NVIDIA drivers for your GPU:
- **Windows**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **Linux**: 
  ```bash
  # Ubuntu/Debian
  sudo apt update
  sudo apt install nvidia-driver-535
  
  # Check installation
  nvidia-smi
  ```

### 2. CUDA Toolkit
Install CUDA 11.8 or 12.1:
- **Windows**: Download from [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- **Linux**:
  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb
  sudo apt-get update
  sudo apt-get -y install cuda-11-8
  ```

### 3. Python Environment
Python 3.10 or 3.11 recommended:

**Windows:**
```powershell
# Download from python.org or use winget
winget install Python.Python.3.11

# Verify
python --version
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/jschroeder-mips/llm_training_mlx.git
cd llm_training_mlx
```

### 2. Create Virtual Environment
**Windows:**
```powershell
python -m venv venv_cuda
.\venv_cuda\Scripts\activate
```

**Linux:**
```bash
python3 -m venv venv_cuda
source venv_cuda/bin/activate
```

### 3. Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA installation:**
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### 4. Install Training Dependencies
```bash
pip install transformers>=4.34.0 peft>=0.5.0 datasets>=2.14.0
pip install bitsandbytes>=0.41.0 accelerate>=0.23.0
pip install sentencepiece protobuf python-dotenv
```

**Windows Note**: If `bitsandbytes` installation fails on Windows, use:
```powershell
pip install bitsandbytes-windows
```

### 5. Set Up Hugging Face Authentication

**Option A: Environment Variable**
```bash
# Linux/Mac
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

**Option B: .env File** (Recommended)
Create `.env` in the repo root:
```
HF_TOKEN=hf_your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## Training

### Run Training Script
```bash
python train_cuda.py
```

**What happens:**
1. Checks GPU availability
2. Downloads dataset (~1MB)
3. Downloads Mistral-7B model (~14GB, first run only)
4. Applies 4-bit quantization and LoRA
5. Trains for 600 iterations (~15-30 minutes)
6. Saves adapters to `./adapters_cuda/final/`

**Expected output:**
```
==============================================================
CUDA/PyTorch LoRA Fine-Tuning for RISC-V Assembly
==============================================================

✓ GPU detected: NVIDIA GeForce RTX 4090
✓ VRAM available: 24.0 GB

==============================================================
STEP 1: Preparing Dataset
==============================================================

Loading tokenizer: mistralai/Mistral-7B-Instruct-v0.3
✓ Loaded 590 examples
...
```

### Training Output
The script shows progress every 50 steps:
```
Step 50:  {'loss': 2.145, 'learning_rate': 1e-05, 'epoch': 0.17}
Step 100: {'loss': 1.523, 'learning_rate': 1e-05, 'epoch': 0.34}
...
Step 600: {'loss': 0.944, 'learning_rate': 1e-05, 'epoch': 2.03}
```

Loss should decrease from ~4.0 to ~1.0.

### Memory Usage
- **RTX 3090/4090 (24GB)**: Uses ~14-16GB with batch_size=1, gradient_accumulation=2
- **Smaller GPUs (16GB)**: Reduce batch size or LoRA rank if OOM occurs

## Inference

### Interactive Testing
```bash
python inference_cuda.py
```

Or specify adapter path:
```bash
python inference_cuda.py --adapter_path ./adapters_cuda/final
```

**Example session:**
```
✓ Using GPU: NVIDIA GeForce RTX 4090
Loading base model: mistralai/Mistral-7B-Instruct-v0.3
✓ Model loaded successfully

Your query: Adds the values in rs1 and rs2, stores the result in rd

Generating response...
============================================================
add rd, rs1, rs2
============================================================
```

## Troubleshooting

### CUDA Not Available
```
⚠️ WARNING: CUDA not available
```

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version
4. Reboot after driver installation

### Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `train_cuda.py`:
   ```python
   per_device_train_batch_size=1,  # Was 2
   gradient_accumulation_steps=4,  # Was 2
   ```

2. Reduce LoRA rank:
   ```python
   lora_config = LoraConfig(
       r=8,  # Was 16
       lora_alpha=8,  # Was 16
   ```

3. Reduce max sequence length:
   ```python
   max_length=256,  # Was 512
   ```

4. Enable gradient checkpointing (slower but uses less memory):
   ```python
   TrainingArguments(
       gradient_checkpointing=True,
   ```

### bitsandbytes Not Working on Windows
```
ImportError: cannot import name 'functional' from 'bitsandbytes'
```

**Solution:**
```powershell
pip uninstall bitsandbytes
pip install bitsandbytes-windows
```

### Training is Very Slow
**Causes:**
- Running on CPU (check `torch.cuda.is_available()`)
- GPU utilization low (check `nvidia-smi`)
- Disk I/O bottleneck (model loading)

**Solutions:**
1. Verify GPU is being used:
   ```python
   watch -n 1 nvidia-smi  # Linux
   # Or check Windows Task Manager → Performance → GPU
   ```

2. Move model cache to SSD:
   ```bash
   # Linux/Mac
   export HF_HOME="/path/to/fast/ssd"
   
   # Windows
   $env:HF_HOME = "D:\fast\ssd\huggingface"
   ```

3. Check batch size isn't too small (should be 1-2)

### Training Loss Not Decreasing
**Check:**
1. Verify LoRA was applied:
   ```
   Trainable parameters: 20,971,520 (0.289%)
   ```
   If 0.000%, LoRA didn't apply correctly.

2. Check learning rate isn't too low (should be 1e-5)

3. Verify data formatting is correct:
   ```python
   print(train_dataset[0])
   # Should show tokenized chat format
   ```

## Performance Comparison

| GPU | VRAM | Training Time | Batch Size | Notes |
|-----|------|---------------|------------|-------|
| RTX 4060 Ti 16GB | 16GB | ~60 min | 1 | Tight fit, may need rank=8 |
| RTX 3090 | 24GB | ~25 min | 1 + grad_accum=2 | Good for learning |
| RTX 4090 | 24GB | ~18 min | 1 + grad_accum=2 | Excellent |
| A6000 | 48GB | ~15 min | 2 + grad_accum=1 | Professional |
| A100 (40GB) | 40GB | ~12 min | 2 + grad_accum=1 | Cloud/datacenter |
| A100 (80GB) | 80GB | ~12 min | 4 | Can use larger batches |

## Multi-GPU Training

If you have multiple GPUs, you can use `accelerate`:

```bash
# Configure accelerate (one-time)
accelerate config

# Run with accelerate
accelerate launch train_cuda.py
```

This automatically distributes training across all available GPUs.

## Saving to Hugging Face Hub

Share your adapters with others:

```python
from huggingface_hub import HfApi

# In train_cuda.py after training
model.push_to_hub("your-username/mistral-7b-riscv-lora")
tokenizer.push_to_hub("your-username/mistral-7b-riscv-lora")
```

Then others can use:
```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    "your-username/mistral-7b-riscv-lora"
)
```

## Comparing MLX vs CUDA

| Aspect | MLX (Apple Silicon) | CUDA (NVIDIA) |
|--------|---------------------|---------------|
| **Platforms** | macOS only | Windows, Linux |
| **Hardware** | M1/M2/M3/M4 | NVIDIA GPUs |
| **Memory** | Unified (shared CPU/GPU) | Dedicated VRAM |
| **Speed** | ~25 min (M4) | ~15-25 min (RTX 3090/4090) |
| **Setup** | Simpler (uv run) | More complex (CUDA drivers) |
| **Quantization** | Built-in | Requires bitsandbytes |
| **Multi-GPU** | N/A | Supported |
| **Cost** | Mac hardware | GPU rental ~$0.20-2/hr |

Both produce equivalent results; choice depends on your hardware.

## Next Steps

1. **Experiment with hyperparameters**: Try different LoRA ranks, learning rates
2. **Fine-tune other models**: Replace `MODEL_NAME` with Llama, Qwen, etc.
3. **Custom datasets**: Adapt for your own domain
4. **Deploy**: Integrate into applications via API (FastAPI, Flask)

## Resources

- **PEFT Documentation**: https://huggingface.co/docs/peft
- **Transformers**: https://huggingface.co/docs/transformers
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **This Repo**: https://github.com/jschroeder-mips/llm_training_mlx
