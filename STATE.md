# Current Technical State – Nov 23rd, 2025

## What's 100% working right now
- data/input.txt → tiny Shakespeare (1.1M chars)
- models/week1/bigram.py → trained, val loss 2.35
- models/week1/mlp.py     → trained today, 415k params, val loss 2.49

## Latest metrics (MLP run today)
- step 0    → val loss 4.16
- step 4500 → val loss 2.49
- generation already respects line breaks + dialogue markers

## Next step (tomorrow)
Single-head causal self-attention → expect val loss ≤ 2.2

No hype. No resume. Just facts.