import torch
import torch.nn as nn

texts = [
"free money now",
"win cash prize",
"claim your reward",
"limited time offer",
"special offer today",
"meeting tomorrow",
"project update",
"schedule a call",
"project status update"
]

labels = [1,1,1,1,1,0,0,0,0]

# ---- build vocabulary ----

words = set()
for t in texts:
    words.update(t.split())

vocab = {w:i+1 for i,w in enumerate(words)}   # 0 reserved for padding
index_to_word = {i:w for w,i in vocab.items()}

def encode(text):
    return [vocab[w] for w in text.split()]

X = [encode(t) for t in texts]
max_len = max(len(x) for x in X)

for i in range(len(X)):
    X[i] = X[i] + [0]*(max_len-len(X[i]))

X = torch.tensor(X)
y = torch.tensor(labels).float().view(-1,1)

vocab_size = len(vocab)+1

# ---- model components ----

embedding = nn.Embedding(vocab_size, 8)

model = nn.Sequential(
    embedding,
    nn.Flatten(),
    nn.Linear(max_len*8,16),
    nn.ReLU(),
    nn.Linear(16,1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# words to monitor
watch_words = ["offer", "free", "project", "meeting"]

# ---- training loop ----

for epoch in range(300):

    pred = model(X)
    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:

        print("\nEpoch:", epoch, "Loss:", loss.item())

        weights = embedding.weight.data

        for w in watch_words:
            idx = vocab[w]
            vec = weights[idx].numpy()
            print(w, vec)

# ---- prediction helper ----

def predict(text):

    x = encode(text)
    x = x + [0]*(max_len-len(x))
    x = torch.tensor([x])

    prob = model(x).item()
    return prob

print("\nPredictions")
print("free cash offer:", predict("free cash offer"))
print("project meeting update:", predict("project meeting update"))