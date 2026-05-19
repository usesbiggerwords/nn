import numpy as np
from jinja2.filters import do_xmlattr


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

lr = 0.01

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
                                       size=(seq_len, vector_dim))

# print(embedding)

Wq = np.random.randn(vector_dim, vector_dim) * 0.1
Wk = np.random.randn(vector_dim, vector_dim) * 0.1
Wv = np.random.randn(vector_dim, vector_dim) * 0.1

Wout = np.random.randn(vector_dim, vocab_size) * 0.1
Bout = np.zeros(vocab_size)

losses = []
for i in range(20000):

    start = np.random.randint(0, len(data) - seq_len - 1)
    token_ids = np.array(data[start:start + seq_len])
    target_ids = np.array(data[start + 1:start + seq_len + 1])


    tok = embedding[token_ids]
    pos = position_embedding

    x = tok + pos

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    scores = (Q @ K.T) / np.sqrt(vector_dim)
    weights = softmax(scores)
    attention = weights @ V

    logits = attention @ Wout + Bout
    probs = softmax(logits)

    # print(probs)

    loss = -np.mean(
        np.log(
            probs[np.arange(seq_len), target_ids] + 1e-9
        )
    )

    # print(loss)
    losses.append(loss)

    dLogits = probs.copy()
    dLogits[np.arange(seq_len), target_ids] -= 1
    dLogits /= seq_len

    dWout = attention.T @ dLogits
    dBout = np.sum(dLogits, axis=0)

    # attention = weight @ V
    dAttention = dLogits @ Wout.T
    dWeights = dAttention @ V.T
    dV = weights.T @ dAttention
    dWv = x.T @ dV

    # weights = softmax(scores)
    # dWeights = d(softmax(scores)) <- the ugly jacobian
    dScores = weights * (
            dWeights - np.sum(dWeights * weights, axis=-1, keepdims=True)
    )
    # scores = (Q @ K.T) / sqrt(vector_dim)
    dScores /= np.sqrt(vector_dim)

    dQ = dScores @ K
    dK = dScores.T @ Q

    # Q = x @ Wq
    dWq = x.T @ dQ

    # K = x @ Wk
    dWk = x.T @ dK

    # x contributes to Q, K, and V
    # so dQ, dK, and dV all contribute to dx
    # x = tok + pos
    # dx = dtok + dpos
    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    dTok = dx
    dPos = dx


    Wv -= lr * dWv
    Wk -= lr * dWk
    Wq -= lr * dWq
    Wout -= lr * dWout
    Bout -= lr * dBout

    dEmbedding = np.zeros_like(embedding)
    for j, token_id in enumerate(token_ids):
        dEmbedding[token_id] += dTok[j]

    dPosition = np.zeros_like(position_embedding)
    for k in range(seq_len):
        dPosition[k] += dPos[k]

    embedding -= lr * dEmbedding
    position_embedding -= lr * dPosition

    if i % 100 == 0:
        print(i, np.mean(losses[-100:]))