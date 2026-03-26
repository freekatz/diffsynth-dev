# Wan Examples (Local Models Only)

These examples are configured for local checkpoint usage.

- Python scripts use `ModelConfig(skip_download=True, ...)`.
- Training shell scripts export:
  - `DIFFSYNTH_SKIP_DOWNLOAD=true`
  - `DIFFSYNTH_MODEL_BASE_PATH=./models`

Prepare your checkpoints under `./models` (or your custom `DIFFSYNTH_MODEL_BASE_PATH`).
