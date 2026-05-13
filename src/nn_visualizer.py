import numpy as np
import matplotlib.pyplot as plt

# weights for two neurons
w1 = np.array([1.0, -0.5])
b1 = -0.2

w2 = np.array([-0.7, 1.0])
b2 = -0.1

# grid of points in input space
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)

X, Y = np.meshgrid(x, y)

# neuron activations
Z1 = w1[0]*X + w1[1]*Y + b1
Z2 = w2[0]*X + w2[1]*Y + b2

A1 = np.maximum(0, Z1)
A2 = np.maximum(0, Z2)

plt.figure(figsize=(6,6))

# plot activation regions
plt.contourf(X, Y, A1 > 0, alpha=0.3)
plt.contourf(X, Y, A2 > 0, alpha=0.3)

# draw decision lines
plt.contour(X, Y, Z1, levels=[0])
plt.contour(X, Y, Z2, levels=[0])

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("ReLU neurons partitioning input space")

plt.show()