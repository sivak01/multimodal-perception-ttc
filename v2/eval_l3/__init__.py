"""
v2/eval_l3 — a third evaluation layer on top of v2/evaluate.py (lock-once)
and v2/evaluate_matched.py (per-sample greedy, no-lock). Does not modify
either of those, central_tracker.py, or any v1 notebook -- see
config_l3.py's module docstring for the session's own ground rules.
"""
