# Data availability

The released model used 499 training examples and 88 validation examples. They
cover Taj Mahal architecture, construction, economics, gardens, Shah Jahan and
Mumtaz Mahal, Mughal succession, conservation, and material culture.

The original source passages and training rows are not redistributed here. The
project did not establish redistribution rights for every source. Frozen
evaluation questions, answer keys, model responses, and grading packets are also
kept private so that future comparisons remain useful.

The expected MLX chat-data format is JSON Lines:

```json
{"messages":[{"role":"system","content":"You are a history tutor..."},{"role":"user","content":"<question>"},{"role":"assistant","content":"<source-checked answer>"}]}
```

For each row, the private audit table also recorded a stable item ID, topic,
fact-family ID, claim IDs, source URLs, review status, and split family. The
dataset builder grouped question variants before splitting, rejected close
copies of evaluation wording, checked token length, and created a fixed training
schedule. Underlying facts intentionally overlapped between training and
evaluation; exact questions did not.

See [Dataset method](../docs/DATASET_METHOD.md) for the teacher/critic workflow
and release rules.

