# Copilot Instructions

## Project Snapshot
- Repository fine-tunes `mistralai/Mistral-7B-Instruct-v0.3` into a RISC-V code assistant on Apple Silicon.
- Training uses MLX/Metal; PyTorch CUDA guidance is irrelevant here.
- Core automation lives in `train_mlx.py`; there are no secondary modules to keep in sync.

## Tooling & Environment
- Install dependencies with `pip install mlx mlx-lm datasets`; nothing else is required for baseline runs.
- Always call `mlx_lm.load` with `tokenizer_config={"trust_remote_code": True}` to load tokenizer safely.
- Keep workflows CPU/GPU-agnostic beyond Metal; do not introduce CUDA, bitsandbytes, or PEFT usage.

## Data Preparation Pattern
- Dataset source is `davidpirkl/riscv-instruction-specification`; fields `description` → prompt, `instructions` → target.
- `tokenizer.apply_chat_template` produces the ChatML-style string; training expects each JSONL line as `{ "text": full_conversation }`.
- Script splits data 90/10 into `train.jsonl` and `valid.jsonl`; respect this convention so evaluation stays predictable.

## Model & Training Workflow
- Training uses `mlx_lm.lora` via subprocess to ensure stability and correct memory management.
- LoRA adapters target all linear layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with `r=16`, `alpha=16`.
- Training uses `max_seq_length=512`, `batch_size=4`, `iters=600`, `learning_rate=1e-5`, `steps_per_eval=50`.
- Adapters are persisted to `adapters.npz`; downstream code must pass `adapter_path=ADAPTER_FILE` when reloading.

## Inference & Validation
- Quick smoke test runs immediately after training via `mlx_lm.generate` with `add_generation_prompt=True`.
- User-facing prompts follow the pattern `Write the RISC-V assembly instruction for...`; maintain the phrasing when building new evals.

## Troubleshooting Hints
- If memory pressure appears, first lower `batch_size`, then LoRA rank; avoid touching quantization (handled internally by MLX).
- Dataset downloads rely on `datasets.load_dataset`; network or auth failures should be handled by guiding users to Hugging Face login rather than reimplementing loaders.
- Hugging Face caching means reruns reuse local data; delete `~/.cache/huggingface` only when data corruption is suspected.

## File Landmarks
- `train_mlx.py` encapsulates the entire pipeline: dataset prep, LoRA injection, training, and inference sanity check.
- `AGENT.md` captures high-level goals and hardware constraints; align any new guidance with its statements.
- Expect artifacts `train.jsonl`, `valid.jsonl`, and `adapters.npz` to appear in the repo root after training; scripts should reference these paths explicitly.
