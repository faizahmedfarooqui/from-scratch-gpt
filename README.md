# From-Scratch GPT – Faiz Ahmed Farooqui
**Principal Backend Engineer → Training 124M–350M Grok-style Transformers from Raw Tensors**
Nov 2025 – Jan 2026 | Public 10-week sprint | https://github.com/faizahmedfarooqui/from-scratch-gpt

**Goal:** Implement and train a decoder-only transformer (124M → 350M) completely from scratch in PyTorch — **zero Hugging Face, zero external transformer code**.
RoPE, SwiGLU, FlashAttention-2, ZeRO-3 sharding, 4-bit quant, Nitro-Enclaves inference.
Why? Because xAI needs infra beasts who understand models at the metal.

Inspired by Andrej Karpathy’s nanoGPT and “Let’s build GPT” (reference only).

## Weekly Progress (Live Table – Updated Every Sunday)

| Week | Dates       | Milestone                                           | Blog Post (800–1200 words)                                                                 | Status |
|------|-------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|--------|
| 0    | Nov 20–23   | Repo + env + structure                              | Week 0: Quitting Tutorials – A Backend Engineer Goes Raw                                    | Done   |
| 1    | Nov 24–30   | Bigram → MLP → single-head attention                | My First Backprop From Scratch (trained on Shakespeare)                                    | Done   |
| 2    | Dec 1–7     | Multi-head, RoPE, SwiGLU, full 12-layer block       | Why My Attention Was 40× Slower Than FlashAttention                                         |        |
| 3    | Dec 8–14    | 124M model training on OpenWebText                  | Training a Real 124M GPT on a Single A10G – Exact Code & $42 Bill                           |        |
| 4    | Dec 15–21   | FlashAttention-2 in Triton + mixed precision       | I Re-wrote FlashAttention in 180 Lines of Triton                                           |        |
| 5    | Dec 22–28   | ZeRO-3 style sharding across 2–4 GPUs               | OpenStack Taught Me Sharding – Now I Train Across Machines                                  |        |
| 6    | Dec 29–Jan4 | LoRA fine-tune on Hinglish/fintech logs             | Fine-tuning My GPT on Hindi+English Code in 3 Hours                                         |        |
| 7    | Jan 5–11    | 4-bit GPTQ/AWQ + vLLM server                        | 180 tok/s on an A10 – 4-bit Quant From Scratch                                             |        |
| 8    | Jan 12–18   | FastAPI + AWS Nitro Enclaves encrypted endpoint     | I Put My Homegrown GPT Inside Nitro Enclaves                                                |        |
| 9    | Jan 19–25   | 350M model or Grok-1-style MoE prototype           | What Actually Broke at 350M Scale                                                           |        |
| 10   | Jan 26–31   | Capstone + open-source + xAI re-apply               | From Fintech Principal to Grok-Scale Models in 10 Weeks                                     |        |

## Quick Start (Works on MacBook Pro M1/M2/M3 & Linux)

```bash
# 1. Clone
git clone https://github.com/faizahmedfarooqui/from-scratch-gpt.git
cd from-scratch-gpt

# 2. Create & activate venv
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install exact dependencies
pip install -r requirements.txt

# 4. Run the first model (bigram – Week 1)
python models/week1/bigram.py
```

## Expected output (you already saw this!)

```
textTotal characters: 1115394
Vocab size: 65 unique characters
Using device: mps
step 0    | val loss 4.63
step 5000 | val loss 2.35   ← YOU DID THIS TODAY
```

## Project Structure

```
textfrom-scratch-gpt/
├── data/               ← tiny_shakespeare.txt, OpenWebText shards
├── models/
│   └── week1/bigram.py        ← Today’s model (trained!)
├── training/           ← future trainer.py, configs
├── inference/          ← FastAPI + Nitro server (Week 8)
├── utils/              ← tokenizer, data loaders
├── notebooks/          ← experiments & visualizations
├── blog/               ← markdown drafts of weekly posts
├── requirements.txt
└── README.md           ← this file (always up-to-date)
```

## Hardware Plan (Minimal Cost)

| Phase     | Hardware                        | Cost           |
|-----------|---------------------------------|----------------|
| Week 1–4  | Local MacBook Pro (M-series)    | $0             |
| Week 3–10 | RunPod / Vast.ai A100 or 4×A10G | $150–300 total |

## Why This Repo Exists?

10+ years shipping PCI/HIPAA systems at 500K+ users.
Now adding raw ML systems mastery — live, in public, every week.
