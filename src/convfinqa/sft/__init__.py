"""Supervised fine-tuning of an open student model on Sonnet 5.5's conversations (s23).

`splits` cuts the `sft_v1` manifest by report page; `filter` decides which
answered conversations are clean enough to train on; `cuts` and `report` build
the train-set review page.
"""
