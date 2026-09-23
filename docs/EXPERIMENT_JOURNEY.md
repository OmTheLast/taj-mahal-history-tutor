# Experiment journey

This pilot began as a small rehearsal for a future Indian-history tutor. The
goal was deliberately narrow: make a 4B model better at answering questions
about the Taj Mahal, and learn which data and evaluation practices survive
contact with real training runs.

## 1. The first corpus was large only on paper

An early bank contained 4,391 retained rows, but the useful count was smaller:
many rows repeated sources, facts, or templates. Treating row count as knowledge
coverage overstated the dataset. The project rebuilt a reviewed release from 859
candidates and retained 499 training plus 88 validation examples.

**Solution:** count unique questions, fact families, source claims, and actual
training presentations separately. Preserve a quarantine list instead of
silently deleting rejected material.

## 2. Leaving examples out did not always mean leaving facts out

For history, an evaluation example can contain a fact that never appears in
training. That tests unseen knowledge rather than whether the training recipe
made a taught fact usable. The opposite problem is also possible: differently
worded questions can leak nearly identical evaluation wording.

**Solution:** separate question overlap from fact overlap. Group variants before
splitting, keep validation facts represented in training for this recipe test,
and hold exact evaluation questions and outputs outside training. Label genuinely
unseen claims separately.

## 3. Lower loss did not equal better historical answers

The 375-update run reached its best validation loss at step100, while later
checkpoints sometimes did better on generated factual answers. The final
checkpoint recalled training examples strongly but did not consistently improve
new formulations.

**Solution:** save checkpoints, freeze the evaluation and rubric before running
them, grade answers anonymously, and select using behavioral gates rather than
training loss alone. Never search many checkpoints on the final test and call the
winner blind.

## 4. Facts were present but unstable across wording

Audits found exact training witnesses for the targeted facts, yet the model could
answer a direct question and fail a paraphrase, relation, or reversed claim.
Missing fact coverage was therefore not a complete explanation.

**Solution:** compare matched direct and varied practice. Across three seeds,
varied questions beat repeated direct questions by 10–16 answers out of 64 while
both arms retained 16/16 exact training prompts.

## 5. Narrow repairs moved errors around

The upkeep-waqf experiments exposed a recurring problem: the model could remember
“village revenue” and “Taj Ganj shops and caravanserais” yet assign the receipts
to the wrong source. Explicit structure helped some mapping questions. Diverse
answers helped false-premise corrections. Neither intervention generalized
reliably across relation and partial-cue questions, and some older answers
regressed.

**Solution:** treat each intervention as a controlled trial. Hold exposure,
schedule, initialization, and seeds fixed; freeze new diagnostics first; keep
broad regression checks; and refuse promotion when gains do not reproduce.

## 6. Grading was itself a source of error

Automated or model-assisted grading sometimes accepted answers that omitted a
required component. One fresh evaluation needed manual review of anonymous
samples, explicit handling of omitted grader rows, and coordinator overrides
before model identities were opened.

**Solution:** grade deduplicated anonymous `(question, answer)` packets, freeze
decisions before unmasking models, manually inspect uncertain outputs and a
preselected sample, and report grader limitations. Keep the scoring rule stable
even when a candidate misses by one item.

## 7. Quantization changed exact outputs

The merged Transformers model reloaded cleanly, but GGUF runtimes did not always
produce the same text. In an eight-prompt runtime smoke test, full-precision GGUF
and Q6_K each exactly matched the MLX answer on 3/8 prompts; Q4_K_M matched 1/8.
This was not a factual score.

**Solution:** publish Q6_K as an explicitly experimental runtime variant, retain
the merged Transformers model as the reference export, and distinguish export
parity from historical correctness.

## 8. The final promotion gate did its job

Hybrid step250 gained 14–21 focused answers over broad-only controls, but one seed
lost five broad answers where the frozen limit allowed four. The difference was
small and grading had limitations, yet changing the threshold afterward would
invalidate the experiment.

**Solution:** keep the released step100 checkpoint, publish the later result as
useful evidence, and require a new independently frozen evaluation before any
future promotion.

## Main lessons for the larger history model

- Measure unique facts and practice operations, not raw row counts.
- Vary question forms while preserving source-checked answer content.
- Use multiple seeds for dataset comparisons.
- Save complete optimizer, scheduler, RNG, and data-progress state.
- Freeze questions, rubrics, thresholds, and checkpoint candidates before
  inference.
- Track paired gains and regressions; a higher total can conceal lost knowledge.
- Audit the grader as carefully as the model.
- Keep source rights and evaluation secrecy in the release plan from day one.
- Expand from the Taj pilot only after this procedure works with independent
  subject-matter review and broader curriculum coverage.

