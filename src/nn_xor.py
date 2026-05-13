import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# XOR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,0])

# weights
w1 = np.random.randn(2,2)
b1 = np.zeros(2)

w2 = np.random.randn(2)
b2 = 0

lr = 0.1

for step in range(5000):

    loss = 0

    for i in range(4):

        inputs = X[i]
        target = y[i]

        # forward
        z1 = np.dot(inputs, w1) + b1
        hidden = relu(z1)

        pred = np.dot(hidden, w2) + b2

        err = pred - target
        loss += err**2

        # backward
        dL_dpred = 2*err

        dL_dw2 = hidden * dL_dpred
        dL_db2 = dL_dpred

        dL_dhidden = w2 * dL_dpred
        dL_dz1 = dL_dhidden * relu_derivative(z1)

        dL_dw1 = np.outer(inputs, dL_dz1)
        dL_db1 = dL_dz1

        # update
        w2 -= lr*dL_dw2
        b2 -= lr*dL_db2

        w1 -= lr*dL_dw1
        b1 -= lr*dL_db1

    if step % 500 == 0:
        print("step", step, "loss", loss)

print("\nResults:")
for x in X:
    h = relu(np.dot(x,w1)+b1)
    p = np.dot(h,w2)+b2
    print(x, "->", round(p,3))