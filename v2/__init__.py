"""v2 — asynchronous measurement-level (early) fusion pipeline.

Parallel implementation alongside the existing 17-notebook late-fusion
pipeline (see ../CLAUDE.md and ../v2_architecture_brief.md). Does not
modify or depend on any existing notebook at import time; adapters.py
(not yet built) is the only module that reads their on-disk outputs.
"""
