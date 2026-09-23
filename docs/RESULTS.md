# Results and model selection

## Released checkpoint

The released model is the validation-selected **step100** adapter merged into
Qwen3-4B-Instruct-2507. It had seen 400 of the 499 unique training examples once,
covering 153 claim IDs and 16,681 supervised assistant tokens.

On the frozen 96-question Taj pilot benchmark:

| Section | Base model | Released model |
|---|---:|---:|
| Multiple choice | 24/32 | 26/32 |
| Short factual answers | 6/32 | 16/32 |
| Reworded factual questions | 4/16 | 6/16 |
| Supplied-note questions | 16/16 | 12/16 |
| **Total** | **50/96** | **60/96** |
| Open factual subtotal | 10/48 | 22/48 |

The adapter gained 19 answers and lost 9. It improved factual recall but reduced
performance on questions that required using only a supplied note. The benchmark
was inspected during development and grading was conducted within the project,
so these figures are pilot evidence rather than an independent educational
evaluation.

## Why step100 was released

The original run lasted 375 updates. Validation loss was lowest at step100 and
then rose while training loss continued to fall:

| Update | Validation loss |
|---:|---:|
| 0 | 4.230 |
| **100** | **1.550** |
| 200 | 1.570 |
| 300 | 1.748 |
| 375 | 1.829 |

Later generated-answer tests showed that loss alone was an imperfect selector:
steps 150 and 200 sometimes answered more validation questions correctly. But
those investigations used development sets and did not establish a replacement
under a frozen promotion gate. The conservative release therefore remains the
checkpoint selected before its release benchmark.

## Question-variety experiments

The strongest repeatable recipe finding was that varied practice improved fresh
formulations while exact training recall stayed equal:

| Seed | Direct /64 | Varied /64 | Advantage |
|---:|---:|---:|---:|
| 42 | 46 | 62 | +16 |
| 43 | 48 | 61 | +13 |
| 44 | 49 | 59 | +10 |

These 64 questions cover 16 correlated fact families, so they must not be read as
64 independent historical facts or as a curriculum-wide score.

## Early stopping experiments

For a later balanced recipe, checkpoint 168 reached 39/48 in both seeds and
matched the final step216 on a one-time locked 16-question confirmation set:
12/16 for seed43 and 10/16 for seed44. This supports a broad performance plateau
and a possible 22% training reduction. It does not prove a unique optimal step.

## Fresh step250 gate

A later comparison froze 192 new questions before inference. On the 184
trained-claim items, hybrid step250 improved focused questions substantially but
missed the predeclared broad-retention rule in one seed:

| Seed | Broad-only total | Hybrid250 total | Broad delta | Focused delta |
|---:|---:|---:|---:|---:|
| 43 | 70/184 | 88/184 | -3 | +21 |
| 44 | 80/184 | 89/184 | **-5** | +14 |

The allowed broad loss was at most four in each seed. Seed44 lost five, so the
checkpoint was not promoted. These preliminary core scores used anonymous
assisted grading with manual audits; the separate clear-false-assertion audit was
not required to overturn a gate that had already failed.

## What the scores mean

The model learned useful Taj facts, but access to those facts remained sensitive
to wording, relations, and misleading premises. Aggregate gains could hide
paired regressions. Every promotion decision therefore used a frozen rule and
kept the older selected checkpoint when any required condition failed.

