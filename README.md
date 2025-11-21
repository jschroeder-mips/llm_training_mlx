# llm_training_mlx

Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` into a RISC-V code assistant using [MLX](https://github.com/ml-explore/mlx) on Apple Silicon or other Metal-capable hardware. The workflow relies on [uv](https://github.com/astral-sh/uv) for environment management and executes entirely through the single `train_mlx.py` script.

## Prerequisites
- Python 3.11+
- A Hugging Face account with access tokens configured
- Sufficient RAM/VRAM (tested on Apple Silicon with 16 GB memory)

Install uv using the official bootstrap command for your platform, then ensure it is on your `PATH`.

### macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Add `~/.local/bin` to your `PATH` if prompted, then verify:

```bash
uv --version
```

### Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Export the path for the current shell session (and add it to your shell profile for persistence):

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```
Close and reopen PowerShell, then verify:

```powershell
uv --version
```

## Hugging Face Authentication
1. Create a Hugging Face [access token](https://huggingface.co/settings/tokens) with `Read` scope.
2. Log in using `uv run huggingface-cli login` and follow the prompt, or export the token:
	- macOS/Linux: `export HF_TOKEN="hf_your_token"`
	- Windows PowerShell: `$env:HF_TOKEN = "hf_your_token"`
3. (Optional) Persist the token across sessions:
	- macOS/Linux: add the export line to `~/.zshrc` or `~/.bashrc`
	- Windows: `setx HF_TOKEN "hf_your_token"`

Once authenticated, uv will download models and datasets through the Hugging Face cache (`~/.cache/huggingface`).

## Setup
1. Clone the repository.
2. In the project root, initialize the uv virtual environment:
	```bash
	uv sync
	```
	This ensures uv records the interpreter version and prepares the cache for script execution. Dependencies are resolved dynamically from the PEP 722 header inside `train_mlx.py`.

## Training Workflow
Run the training script through uv so dependencies are isolated and reproducible:

```bash
uv run train_mlx.py
```

The script performs the following steps automatically:
- Downloads `davidpirkl/riscv-instruction-specification` from Hugging Face.
- Formats prompts with the Mistral chat template and writes `train.jsonl` / `valid.jsonl`.
- Freezes the base model, applies LoRA adapters to `q_proj` and `v_proj`, and trains for 600 iterations.
- Saves adapters to `adapters.npz` and executes a quick inference smoke test.

Artifacts (`train.jsonl`, `valid.jsonl`, `adapters.npz`) are written to the repository root. Re-run `uv run train_mlx.py` any time to regenerate them.

## Troubleshooting
- **Hugging Face auth errors**: re-run `uv run huggingface-cli login` and ensure the `HF_TOKEN` environment variable is set.
- **Memory pressure**: lower `batch_size` inside the script first, then reduce LoRA rank (`r`) if necessary.
- **Fresh start**: clear the Hugging Face cache at `~/.cache/huggingface` or `%USERPROFILE%\.cache\huggingface` if downloads become corrupted.

## Next Steps
- Load `adapters.npz` with `mlx_lm.load(..., adapter_path="adapters.npz")` to run inference against the fine-tuned assistant.
- Customize dataset splits or evaluation logic by editing `train_mlx.py`.
