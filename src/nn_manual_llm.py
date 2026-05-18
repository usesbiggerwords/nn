import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# initialize

training_data = "hello world " * 100

chars = sorted(list(set(training_data)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

lr = 0.001

# base constraints

vector_dim = 16
vocab_size = len(chars)
seq_len = 8

data = [stoi[c] for c in training_data]

embedding = np.random.uniform(low=-1.0,
                              high=1.0,
                              size=(vocab_size, vector_dim))
position_embedding = np.random.uniform(low=-1.0,
                                       high=1.0,
                                       size=(vocab_size, vector_dim))

# print(embedding)

Wq = np.random.randn(vector_dim, vector_dim) * 0.1
Wk = np.random.randn(vector_dim, vector_dim) * 0.1
Wv = np.random.randn(vector_dim, vector_dim) * 0.1

Wout = np.random.randn(seq_len, seq_len) * 0.1
bout = np.zeros(seq_len)

# one iteration

x = embedding + position_embedding
Q = x @ Wq
K = x @ Wk
V = x @ Wv

scores = (Q @ K.T) / np.sqrt(vector_dim)
weights = softmax(scores)
attention = weights @ V

logits = weights @ Wout + bout
probs = softmax(logits)

print(probs)