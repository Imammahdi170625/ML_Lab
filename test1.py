import numpy as np


def sigmid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
D = np.array([[0],
              [0],
              [1],
              [1],])
W = np.random.rand(1,3)
alpha = .9
for i in range(1000):
    for j in range(D.shape[0]):
        x = np.reshape(X[j], (3, 1))
        d = D[j, :]
        v = np.dot(W, x)
        y = sigmid(v)
        e = d - y
        delta = y * (1 - y) * e
        dW = alpha * delta * x
        # print(np.reshape(dW,(1,3)))
        # W = W + np.reshape(dW,(1,3))
        W = W + np.transpose(dW)

for i in range(D.shape[0]):
    x = np.reshape(X[i],(3,1))
    v = np.dot(W,x)
    y = sigmid(v)
    print(y)