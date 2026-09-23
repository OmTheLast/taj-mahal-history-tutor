# Model card

## Model

**Taj Mahal History Tutor 4B — experimental pilot** is a Taj-specific LoRA
fine-tune of Qwen3-4B-Instruct-2507. The published Transformers version merges
the selected step100 adapter into the base model.

Weights: <https://huggingface.co/OmTheLast/taj-mahal-history-tutor-4b-experimental>

## Intended use

Research and prototyping of short English-language questions about the Taj Mahal.
Verify factual answers against authoritative sources before using them for
teaching or assessment.

## Out of scope

The model is not certified for NCERT, ICSE, state-board, or general Indian
history coverage. It is not a source of record and should not grade students.

## Training and evaluation

The model used 499 source-audited training examples and 88 validation examples.
Step100 was selected by minimum validation loss before the 96-question release
benchmark. It scored 60/96 compared with 50/96 for the base model. Open factual
answers rose from 10/48 to 22/48, while supplied-note performance fell from 16/16
to 12/16.

See [Results](docs/RESULTS.md) for the full interpretation.

## Limitations

- It can omit required parts of multi-part answers.
- Dates, counts, relationships, attributions, and source-specific claims remain
  unstable under rewording.
- It sometimes adds unsupported details.
- The release benchmark was inspected during development and is not an
  independent measure of educational quality.
- The Q6_K export received only a small runtime smoke test and can generate
  different wording from the reference Transformers model.

## License

Apache License 2.0, consistent with the Qwen base model. Users remain responsible
for the terms of any input sources used in their own retraining datasets.

