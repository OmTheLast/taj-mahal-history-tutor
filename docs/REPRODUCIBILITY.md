# Reproducibility notes

## Hardware and training stack

- Apple Mac M4 Max with 64 GB unified memory
- MLX and `mlx-lm` for local LoRA training
- Qwen3-4B-Instruct-2507 as the base model
- Rank-16 LoRA, scale 32, last 16 layers
- Batch size 4, Adam, constant learning rate `2e-5`
- Assistant-suffix-only loss, maximum sequence length 512
- Seed 42 for the released run

Peak MLX memory for the released recipe was about 9.53 GB.

## Repository layout

- `configs/`: the released run configuration
- `scripts/build_taj_recipe.py`: split, overlap, token, and schedule builder
- `scripts/train_taj_recipe.py`: training, validation, checkpoint selection, and
  resumable journaling
- `scripts/checkpoint_state.py`: atomic full-state save and restore helpers
- `scripts/evaluate_taj_recipe.py`: deterministic evaluation runner
- `docs/`: methods, results, journey, and limitations

## What is needed to rerun

1. Download `Qwen/Qwen3-4B-Instruct-2507` into `models/qwen3-4b-base`.
2. Prepare source-cleared JSONL data matching `data/README.md`.
3. Generate and freeze a group-aware split, schedule, manifest, and hashes.
4. Run the checkpoint recovery self-test.
5. Run `python scripts/train_taj_recipe.py`.
6. Use `--stop-after N` to simulate interruption, then rerun the same command to
   resume from the latest complete checkpoint.
7. Evaluate only against an independently frozen benchmark.

The checked-in scripts are an archival snapshot of the selected run. They retain
the original integrity checks and expect private manifests and helper files that
are intentionally absent. This repository therefore documents the exact method;
it is not a one-command claim that the private experiment can be reconstructed
without its source-cleared data and frozen evaluation records.

## Checkpoint contents

Each complete checkpoint stores:

- LoRA weights
- optimizer tensors
- MLX random state
- NumPy random state
- training step and schedule position
- supervised-token count
- scheduler metadata
- best validation record
- file and settings hashes

Writes first go to an `.incomplete` directory and are atomically renamed only
after the files are flushed. On resume, incomplete or hash-invalid checkpoints
are quarantined and the metrics journal is rolled back to the chosen checkpoint.

## Public weights

- Transformers and original MLX adapter:
  `OmTheLast/taj-mahal-history-tutor-4b-experimental`
- Experimental Q6_K GGUF:
  `gguf/taj-step100-Q6_K.gguf` within that Hugging Face repository

