# Taj Mahal History Tutor 4B

[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-FFD21E)](https://huggingface.co/OmTheLast/taj-mahal-history-tutor-4b-experimental)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

An experimental 4B English-language tutor trained to answer questions about the
Taj Mahal. This repository documents the complete research journey: how the data
was prepared, what improved, what failed, how overfitting and checkpoint choice
were investigated, and why later candidates were not promoted.

The released model is a LoRA fine-tune of
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
It is a small recipe-development pilot for a future Indian-history model. It is
not a curriculum-complete school-history model.

## Get the model

| Format | Location | Status |
|---|---|---|
| Merged Transformers safetensors | [Hugging Face](https://huggingface.co/OmTheLast/taj-mahal-history-tutor-4b-experimental) | Reference export |
| Original MLX LoRA adapter | [Hugging Face `mlx-adapter/`](https://huggingface.co/OmTheLast/taj-mahal-history-tutor-4b-experimental/tree/main/mlx-adapter) | Training artifact |
| Q6_K GGUF | [Hugging Face `gguf/`](https://huggingface.co/OmTheLast/taj-mahal-history-tutor-4b-experimental/tree/main/gguf) | Experimental runtime variant |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "OmTheLast/taj-mahal-history-tutor-4b-experimental"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "system",
        "content": (
            "You are a history tutor. Answer each question directly using the "
            "requested format. Give accurate information and do not invent facts."
        ),
    },
    {"role": "user", "content": "Where is the Taj Mahal?"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True))
```

## Main result

On the frozen 96-question Taj pilot benchmark, the base model scored **50/96**
and the released step100 checkpoint scored **60/96**. Open factual questions
improved from **10/48** to **22/48**, while supplied-note questions regressed
from **16/16** to **12/16**.

| Section | Base | Released |
|---|---:|---:|
| Multiple choice | 24/32 | 26/32 |
| Short factual | 6/32 | 16/32 |
| Reworded factual | 4/16 | 6/16 |
| Supplied note only | 16/16 | 12/16 |

The result is useful but limited. The model still makes factual errors, omits
parts of answers, and changes behavior under rewording.

## What we learned

1. **Raw dataset size was misleading.** Unique facts, source claims, question
   families, and actual presentations mattered more than row count.
2. **Question variation mattered.** Across three seeds, varied practice scored
   59–62/64 on fresh formulations, versus 46–49/64 for repeated direct practice.
3. **Loss was not enough to choose a checkpoint.** Later checkpoints sometimes
   answered more facts correctly even after validation loss began rising.
4. **Narrow repairs often moved errors around.** Improving one relation or false
   premise could cause an older wording to regress.
5. **Grading needed auditing.** Anonymous model-assisted grades still required
   manual samples and explicit overrides.
6. **Frozen gates prevented wishful promotion.** A later step250 candidate gained
   strongly on focused questions but lost five broad answers in one seed, one
   beyond the predeclared limit, so it was not released.

Read [the experiment journey](docs/EXPERIMENT_JOURNEY.md) for the difficulties
and solutions, and [the result report](docs/RESULTS.md) for the score tables.

## Repository map

| Path | Purpose |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use and limitations |
| [`docs/EXPERIMENT_JOURNEY.md`](docs/EXPERIMENT_JOURNEY.md) | Chronological difficulties and solutions |
| [`docs/TIMELINE.md`](docs/TIMELINE.md) | Compact record of each experimental round |
| [`docs/DATASET_METHOD.md`](docs/DATASET_METHOD.md) | Luna teacher / Muse critic workflow |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Benchmark and checkpoint results |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Hardware, state recovery, and rerun notes |
| [`configs/taj_recipe_v1.json`](configs/taj_recipe_v1.json) | Released training configuration |
| [`scripts/`](scripts) | Archival data, training, recovery, and evaluation code |
| [`data/README.md`](data/README.md) | Data schema and publication boundary |

## Dataset and evaluation boundary

This repository does not publish source passages, private training rows, frozen
evaluation questions and keys, raw model answers, provider traces, or model
weights. Source redistribution rights were not established for every document,
and keeping evaluation records private preserves their usefulness. The public
model weights live on Hugging Face.

## License and acknowledgement

The code and model are released under Apache License 2.0. The project builds on
Qwen3-4B-Instruct-2507. Luna drafted and revised dataset exercises; Muse Spark
1.3 Free critiqued dataset content; the project coordinator resolved source and
evaluation decisions. Model-generated review was treated as evidence, not as
independent historical certification.
