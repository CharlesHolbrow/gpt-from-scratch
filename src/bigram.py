# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3_000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)

with open("./input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Pseudo data handling
data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

# test
assert ''.join(decode(encode("hello world!"))) == "hello world!"

x = train_data[0: block_size]
y = train_data[1: block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(t, context, "->", target)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Decorator to disable gradient computation
def estimate_loss():
    out = {}
    # Some layers have different behavior during training and evaluation, for
    # example: batch normalization and dropout.
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # .item()? 
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are both (B, T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        logits = self.lm_head(token_embeddings) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions (as logits)
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, 1) # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # At an interval, eval loss on train and val data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_l = losses['train']
        val_l = losses['val']
        print(f"Step {iter}, Train loss: {train_l:.4f}, Val loss: {val_l:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# %%
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500).squeeze().tolist()))


# %%
