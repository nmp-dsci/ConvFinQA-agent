"""The eval loop (s04, M1): splits → run → trace → score → gate → promote.

One package for the loop the s04 plan describes. M1 scope: a committed split
manifest, a runner that makes every pass an MLflow run whose id is stamped on
every trace row, and a paired gate that decides promotion. The teacher
(diagnoser) arrives in M2.
"""
