"""Experiment tracking, bundle versioning, and the promotion contract.

bundle      — what "a model version" means here: the whole prompt bundle
traces      — per-turn stage IO for every serving and eval turn
mlflow_log  — run logging, from inside the runners rather than beside them
comparator  — accuracy floor + no pass->fail flips
registry    — versions, champion/challenger aliases, append-only history
backfill    — reconstruct the history that predates this package
snapshot    — the committed export the keyless demo reads
gate        — the CI entry point
"""

from convfinqa.tracking.bundle import bundle_fingerprint, bundle_id

__all__ = ["bundle_fingerprint", "bundle_id"]
