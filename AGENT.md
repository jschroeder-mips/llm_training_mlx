Project Context: RISC-V LLM Fine-Tuning on Apple Silicon

1. Project Goal

The objective is to fine-tune a Mistral-7B-Instruct-v0.3 model to function as a specialized RISC-V assembly code generator. The model should accept natural language descriptions of operations and output the corresponding valid RISC-V assembly instructions.

2. Environment & Hardware

Machine: MacBook Pro (M4 Chip).

Chip Architecture: Apple Silicon (ARM64).

Compute Platform: Metal (GPU).

Constraint: We are NOT using CUDA (NVIDIA). We are NOT using standard PyTorch MPS for training due to lack of 4-bit quantization support (bitsandbytes is incompatible with Apple Silicon).

3. Software Stack & Framework Decisions

To achieve efficient local fine-tuning on the M4 chip, we have selected Apple MLX over PyTorch.

Framework: mlx and mlx-lm.

Why MLX?

Native support for Unified Memory architecture.

Native support for 4-bit quantization (QLoRA) without requiring CUDA-only libraries like bitsandbytes.

Significantly higher throughput for training on M-series chips compared to torch.device("mps").

4. Dataset Information

Source: Hugging Face - davidpirkl/riscv-instruction-specification

Structure:

description: Natural language prompt (Input).

instructions: RISC-V assembly code (Target Output).

Formatting Strategy: * The raw dataset must be converted to a ChatML-style or text-completion format before training.

Format: User: <description>\nAssistant: <instructions>

5. Implementation Details (train_mlx.py)

The core training logic is encapsulated in train_mlx.py.

Key Configurations

Base Model: mistralai/Mistral-7B-Instruct-v0.3

LoRA Configuration:

Rank (r): 16

Alpha: 16

Target Modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj (all linear layers).

Training Parameters:

Quantization: 4-bit (handled via mlx_lm.load).

Batch Size: 4 (optimized for 16GB+ RAM).

Iterations: 600 (~1 epoch).

Optimizer: Default MLX optimizer (AdamW equivalent).

Workflow

Data Prep: Download dataset, format prompts, and save as train.jsonl and valid.jsonl.

Train: Execute training loop using mlx_lm.lora (via subprocess).

Save: Export adapters to adapters.npz.

Inference: Reload model with adapters and test generation using mlx_lm.generate().

6. Instructions for the AI Agent

When generating code or debugging for this project, follow these rules:

Do not suggest CUDA libraries: Avoid bitsandbytes, auto-gptq, or flash-attention (unless the specific Metal implementation is available).

Use MLX Syntax: Prefer mlx.core and mlx.nn over torch and torch.nn.

Inference: Do not use peft for inference on this local machine. Use mlx_lm.load(..., adapter_path="adapters.npz").

Memory Management: If Out-Of-Memory (OOM) occurs, suggest reducing batch_size or the LoRA rank (r).

7. Reference Commands

# Installation
pip install mlx mlx-lm datasets

# Run Training
uv run train_mlx.py
