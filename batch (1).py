import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0, 0, 1, 1]])
W = np.random.rand(1, 3)
ALPHA = 0.9
dWsum = np.zeros((3, 1))
error_list = []
for i in range(100):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        d = D[:, j]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        if j == 2:
            error_list.append(e[0][0])
        delta = y * (1 - y) * e
        dW = ALPHA * delta * x
        dWsum = dWsum + dW

    dWavg = np.divide(dWsum, 4)
    W = W + np.transpose(dWavg)

print(W)

for i in range(D.shape[1]):
    x = np.reshape(X[i], (3, 1))
    v = np.dot(W, x)
    y = sigmoid(v)
    print(y)

fig, ax = plt.subplots()
fig.suptitle("error graph")
bx = []
for i in range(100):
    bx.append(i)
ax.plot(bx, error_list)
plt.show()