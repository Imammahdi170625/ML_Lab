import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
D = np.array([[0, 1, 1, 0]])
W1 = np.random.rand(4,3)
W2 = np.random.rand(1,4)
alpha = 0.9

for i in range(10000):
    for j in range(X.shape[0]):
        x = np.reshape(X[j], (3, 1))
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v2 = np.dot(W2, y1)
        y2 = sigmoid(v2)
        d = D[0, j]
        e = d - y2
        delta = y2 * (1 - y2) * e
        e1 = np.dot(np.transpose(W2), delta)
        delta1 = y1 * (1 - y1) * e1
        dW1 = alpha * np.dot(delta1, np.transpose(x))
        W1 = W1 + dW1
        dW2 = alpha * np.dot(delta, np.transpose(y1))
        W2 = W2 + dW2

# print(W1)
# print(W2)

input1 = int(input("enter the first value: "))
input2 = int(input("enter the second value: "))
bias = 1
newInput = np.array([[input1,input2,1]])
newV1 = np.dot(W1,np.transpose(newInput))
newy1 = sigmoid(newV1)
newV2 = np.dot(W2,newy1)
final = sigmoid(newV2)
print(final)

