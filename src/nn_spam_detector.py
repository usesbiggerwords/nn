import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

texts = [
"free money now",
"win cash prize",
"claim your reward",
"limited time offer",
"special offer today",
"project status update",
"meeting tomorrow",
"project offer update",
"let's schedule a call"
]

labels = [1,1,1,1,1,0,0,0,0]  # 1 = spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1,1)

model = nn.Sequential(
    nn.Linear(X.shape[1],8),
    nn.ReLU(),
    nn.Linear(8,1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):

    pred = model(X)
    loss = loss_fn(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(epoch, loss.item())

test = "free cash offer"

vec = torch.tensor(vectorizer.transform([test]).toarray(), dtype=torch.float32)

print("spam probability:", model(vec).item())