# models/week1/mlp.py
# MLP Language Model from scratch – Faiz Ahmed Farooqui, Nov 2025
# Evolution of bigram.py: Add hidden layers, ReLU, dropout

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load data (same as bigram)
# ------------------------------------------------------------
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = "data/input.txt"
if not os.path.exists("data"):
    os.makedirs("data")
urllib.request.urlretrieve(url, path)

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Total characters:", len(text))
print(text[:500])

# ------------------------------------------------------------
# 2. Vocab (same)
# ------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ------------------------------------------------------------
# 3. Data split (same)
# ------------------------------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------------------------------------------------
# 4. Hyperparams (tuned for MLP)
# ------------------------------------------------------------
batch_size = 64  # Bigger batch for stability
block_size = 32  # Longer context
max_iters = 5000
eval_interval = 500
learning_rate = 6e-4  # Lower LR for deeper net
eval_iters = 200
n_embd = 128  # Embedding dim (bigger than bigram)
n_hidden = 512  # MLP hidden size
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
dropout = 0.1
print(f"Using device: {device}")

# ------------------------------------------------------------
# 5. Batch function (same, but longer blocks)
# ------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ------------------------------------------------------------
# 6. MLP Model Class
# ------------------------------------------------------------
class MLPModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_hidden, block_size, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Positional encoding
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # MLP layers
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_embd)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        # MLP forward pass
        x = self.layers(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Get last token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------------------------------------
# 7. Training loop (with eval)
# ------------------------------------------------------------
model = MLPModel(vocab_size, n_embd, n_hidden, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"Model params: {sum(p.numel() for p in model.parameters())}")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample batch
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished!")
print("Sample generation:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))