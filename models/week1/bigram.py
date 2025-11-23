# models/week1/bigram.py
# Bigram Language Model from scratch – Faiz Ahmed Farooqui, Nov 2025

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Download tiny Shakespeare dataset (1MB)
# ------------------------------------------------------------
import urllib.request
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = "data/input.txt"
urllib.request.urlretrieve(url, path)

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

print("Total characters:", len(text))
print(text[:500])

# ------------------------------------------------------------
# 2. Build vocabulary (all unique chars)
# ------------------------------------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ------------------------------------------------------------
# 3. Train/val split
# ------------------------------------------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------------------------------------------------
# 4. Hyperparameters
# ------------------------------------------------------------
batch_size = 32
block_size = 8      # context length
learning_rate = 1e-2
steps = 5000
eval_interval = 500
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ------------------------------------------------------------
# 5. Get batch function
# ------------------------------------------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ------------------------------------------------------------
# 6. Simple Bigram Model
# ------------------------------------------------------------
class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C=vocab_size)

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
            logits, _ = self(idx, None)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------------------------------------
# 7. Training loop
# ------------------------------------------------------------
model = BigramModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

for step in range(steps + 1):
    if step % eval_interval == 0:
        # evaluation
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch('val')
            _, val_loss = model(val_x, val_y)
            val_losses.append((step, val_loss.item()))
            print(f"step {step} | val loss {val_loss.item():.4f}")

    # training step
    model.train()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    train_losses.append((step, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished!")
print("Sample generation:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated))