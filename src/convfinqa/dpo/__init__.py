"""DPO round 3 (s33): sample the SFT student on its own history, pair its tries.

`manifest` cuts `dpo_v1` from the fresh pool; `sample` asks each question
several times on one history and stores every try; `pairs` grades the tries
and picks chosen/rejected pairs. The plan is `.lavish/s33_dpo-plan.html`.
"""
