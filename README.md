# from-scratch-gpt

Public repo: training a Grok-style transformer from raw tensors
Faiz Ahmed Farooqui – Nov 2025 → Jan 2026
https://github.com/faizahmedfarooqui/from-scratch-gpt

Zero Hugging Face. Zero external transformer code. Only PyTorch + backprop.

## Current shipped models (Nov 23, 2025)

| Model      | Params  | Val loss | Notes                                      |
|------------|---------|----------|--------------------------------------------|
| Bigram     | ~4k     | 2.35     | Simple embedding lookup                    |
| MLP        | 415k    | 2.49     | Hidden layers + ReLU + learned pos emb     |

Both trained on tiny Shakespeare (1.1M chars) from scratch on MacBook Pro (MPS).

## Quick start

```bash
git clone https://github.com/faizahmedfarooqui/from-scratch-gpt.git
cd from-scratch-gpt

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Bigram model
python models/week1/bigram.py

# MLP model (today)
python models/week1/mlp.py
```

## Project Layout

```
from-scratch-gpt/
├── data/               ← tiny_shakespeare.txt
├── models/
│   └── week1/
│       ├── bigram.py   ← val loss 2.35
│       └── mlp.py      ← val loss 2.49 (today)
├── STATE.md            ← one-line live status (updated daily)
└── README.md           ← this file
```
