import torch
import torch.nn as nn
import torch.optim as optim

text = "hello world " * 100

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

vocab_size = len(chars)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

seq_len = 8

def get_batch():
    i = torch.randint(len(data)-seq_len-1,(1,))
    x = data[i:i+seq_len]
    y = data[i+1:i+seq_len+1]
    return x.unsqueeze(0),y.unsqueeze(0)

class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 32

        self.embed = nn.Embedding(vocab_size,d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=2,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,d_model)
        )

        self.lm_head = nn.Linear(d_model,vocab_size)

        self.last_attn = None

        self.pos_embed = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        B, T = x.shape

        tok = self.embed(x)

        positions = torch.arange(T, device=x.device)
        pos = self.pos_embed(positions)

        x = tok + pos

        attn_out, attn_weights = self.attn(x, x, x)

        self.last_attn = attn_weights

        x = x + attn_out
        x = x + self.ff(x)

        return self.lm_head(x)

model = ToyTransformer()

optimizer = optim.Adam(model.parameters(),lr=0.001)

loss_fn = nn.CrossEntropyLoss()

for step in range(2000):

    x,y = get_batch()

    logits = model(x)

    loss = loss_fn(
        logits.view(-1,vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step,loss.item())

def generate(start,length=40):

    x = torch.tensor([[stoi[c] for c in start]])

    for _ in range(length):

        logits = model(x[:, -seq_len:])

        probs = torch.softmax(logits[0,-1],dim=0)

        next_char = torch.multinomial(probs,1)

        x = torch.cat([x,next_char.unsqueeze(0)],dim=1)

    return "".join(itos[int(i)] for i in x[0])

print(generate("hello"))

import matplotlib.pyplot as plt

def show_attention(text):

    x = torch.tensor([[stoi[c] for c in text]])

    model(x)

    attn = model.last_attn[0].detach().numpy()

    plt.imshow(attn,cmap="viridis")

    plt.xticks(range(len(text)),list(text))
    plt.yticks(range(len(text)),list(text))

    plt.title("Attention Matrix")
    plt.colorbar()

    plt.show()

show_attention("hello wor")