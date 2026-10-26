"""Supervised fine-tuning of an open student model on llm_react_v1's conversations (s28).

`splits` cuts the `sft_v1` manifest by report page (s23). `trajectories` fetches and
reads the open runtime's message histories, `rows` turns one into a training row (the
reasoning kept, the text-answer detour collapsed), `build` gates whole conversations
and writes the dataset, `train` and `cloud` are tau-telecom-sft's Modal recipe, and
`cli` is `convfinqa-sft`.
"""
