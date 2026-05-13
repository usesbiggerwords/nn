import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

target = 10
inputs = np.array([1.0, 2.0, 3.0])

w1 = np.array([[0.2, 0.1],
               [0.8, -0.5],
               [0.5, -0.5]])

w2 = np.array([0.2, 0.1])

b1 = 2.0
b2 = 1.1

lr = 0.001

print(w1)
print(b1)
print(w2)
print(b2)

for step in range(200):

    # ---- forward pass ----
    z1 = np.dot(inputs, w1) + b1
    hidden = relu(z1)

    prediction = np.dot(hidden, w2) + b2

    loss = (prediction - target) ** 2

    # ---- backward pass ----

    # dL/dprediction
    dL_dpred = 2 * (prediction - target)

    # gradients for output layer
    dL_dw2 = hidden * dL_dpred
    dL_db2 = dL_dpred

    # gradient flowing back to hidden layer
    dL_dhidden = w2 * dL_dpred

    # ReLU derivative
    dhidden_dz1 = relu_derivative(z1)

    dL_dz1 = dL_dhidden * dhidden_dz1

    # gradients for first layer
    dL_dw1 = np.outer(inputs, dL_dz1)

    dL_db1 = np.sum(dL_dz1)

    # ---- parameter updates ----

    w2 -= lr * dL_dw2
    b2 -= lr * dL_db2

    w1 -= lr * dL_dw1
    b1 -= lr * dL_db1

    if step % 20 == 0:
        print("step:", step)
        print("prediction:", prediction)
        print("loss:", loss)
        print()

print(w1)
print(b1)
print(w2)
print(b2)