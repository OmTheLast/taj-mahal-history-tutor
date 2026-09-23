# Dataset method

## Scope

This was a controlled Taj Mahal pilot, designed to learn how to build a later
Indian-history model. It was not a curriculum-complete NCERT, ICSE, or state-board
dataset.

The source-audited material covered architecture, construction and cost,
gardens, patronage, Shah Jahan and Mumtaz Mahal, family and succession politics,
later repairs and conservation, and selected objects connected to the court.

## Teacher and critic

The data workflow used two language models in separate roles:

1. **Luna acted as teacher.** It drafted question-answer exercises from the
   approved source ledger and revised items after criticism.
2. **Muse Spark 1.3 Free acted as dataset critic.** It checked relevance,
   ambiguity, source support, answer completeness, unwanted clues, duplicates,
   and wording quality.
3. **The coordinator resolved disagreements.** Critic suggestions were evidence,
   not automatic approval. Source checks overruled unsupported edits.

Neither teacher nor critic graded the target model's answers. This separation
reduced the chance that model outputs would leak into training material.

## Training release

The first selected recipe reviewed 859 candidate exercises. After revisions,
deduplication, overlap checks, token checks, and quarantine, it retained:

| Split | Examples |
|---|---:|
| Training | 499 |
| Validation | 88 |
| Total | 587 |

Question variants from the same family were grouped before splitting. Every
validation claim remained represented in training, because validation measured
new wording for taught Taj facts rather than recall of wholly unseen historical
facts. Exact benchmark wording and all evaluation outputs stayed outside the
training pipeline.

## The most important dataset finding

Repeated direct questions produced high exact training recall but weaker transfer
to reworded questions. A matched trial held the 16 fact groups, answers, exposure,
and supervised-token budget fixed while changing question practice:

| Training style | Fresh questions correct |
|---|---:|
| Repeated direct practice | 46/64 |
| Varied practice | 62/64 |

Two additional seeds reproduced the direction of the effect: varied practice
scored 61/64 versus 48/64, and 59/64 versus 49/64. This supports mixing direct,
paraphrase, relation, confirmation, and false-premise correction exercises.

It did not solve every problem. Narrow repairs around upkeep waqf facts often
improved the exact practiced operation while regressing another wording. The
practical rule is to vary both questions and answer structures, then test fresh
wordings and broad regressions before adopting the change.

## Publication boundary

This repository publishes the method, aggregate results, selected configuration,
and training code. It excludes source passages, the private corpus, evaluation
questions and keys, answer-bearing grading artifacts, provider traces, and model
weights. The weights are published separately on Hugging Face.

