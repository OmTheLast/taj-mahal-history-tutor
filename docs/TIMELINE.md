# Experiment timeline

All dates are in September 2026. Each row records the conclusion at the time;
later work did not silently rewrite earlier gates or promote rejected adapters.

| Date | Round | What changed | Main result | Decision |
|---|---|---|---|---|
| 16 | Source-audited recipe v1 | Rebuilt the corpus through Luna drafting, Muse criticism, coordinator checks, grouping, and quarantine | Base 50/96 → step100 60/96; open facts 10/48 → 22/48 | Release step100 as the pilot; investigate overfitting |
| 16 | Checkpoint comparison | Compared original and steps 50/100/150/200/375 on generated factual validation and exact recall | Steps 150–200 often beat step100 behaviorally; step375 increased recall without equivalent validation gain | Do not replace the selected model from this development comparison |
| 16 | Fresh formulation confirmation | Tested steps 100/150/200 on 64 new wordings across 16 known fact families | 23/64, 27/64, 30/64; unstable reversed-claim corrections | Repair question-operation coverage before more fact collection |
| 16 | Fact coverage audit | Located training witnesses for 32 target propositions | All had training evidence before step100; missing facts were not the full explanation | Test practice form rather than simply adding those facts |
| 16 | Direct vs varied practice | Held fact groups, answers, exposures, and answer-token budget fixed | Direct 46/64; varied 62/64 | Replicate across seeds |
| 16 | Variety replication | Repeated the comparison with seeds 43 and 44 | Varied beat direct by +13 and +10 | Adopt variation as the strongest recipe signal, with limits |
| 17 | Waqf prompt repair | Replaced six problematic prompts | Mixed 18–19/22 results; no stable overall gain | Reject promotion |
| 17 | Answer structure | Made rural and urban revenue mappings explicit | F10 mapping improved, but old and unrelated items regressed | Use as diagnostic evidence only |
| 17 | Answer diversity | Used six answer phrasings and explicit false-premise denial | False-premise corrections improved; relation and partial cues did not | Audit operation coverage before retraining |
| 17 | Operation coverage | Added 24 exercises across six operations | Fresh F10 rose 12→16/24 and 13→18/24; older core slipped | Reject promotion; inspect paired errors |
| 17 | Retention confirmation | Tested saved varied and balanced checkpoints on 72 fresh questions | Earlier-family totals held at 38→38 and 38→39; new families rose sharply | No order-only training; stress-test wording |
| 17 | Formulation stress | Tested difficult dates, uncertainty, relations, and partial cues | Balanced checkpoints trailed varied by 2 and 3 of 32 | Treat wording sensitivity as unresolved |
| 17 | Saved-checkpoint sweep | Evaluated 30 saved checkpoints and one-time locked questions | Balanced step168 matched final216 on locked16 | Early stop is plausible, but not a unique optimum |
| 19–22 | Fresh step250 gate | Froze 192 new questions before seven saved adapters answered them | Hybrid250 gained focused items, but seed44 broad retention was -5 against an allowed -4 | Keep step100; do not move the goalpost |
| 22 | Packaging | Merged step100 and converted GGUF variants | Transformers reload passed; Q6_K selected for experimental runtime publication | Publish weights on Hugging Face with limitations |
| 23 | Research repository | Curated code, aggregate evidence, and lessons without private evaluations or source-restricted data | Repository-level link, syntax, secret, and large-artifact checks passed | Publish the journey separately from weights |

