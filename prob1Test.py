import math

import numpy as np
import matplotlib.pyplot as plt

def bipolar_sigmoid(z):
    return (1 - np.exp(-z))/(1 + np.exp(-z))

def sse(val):
    return np.sum(np.square(np.subtract(np.mean(val)-val)))
input = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [-1, -1, -1, -1]])
# print(input)
target = np.array([[-1,-1,-1,1]])
w = np.random.rand(1,input.shape[0])
ab = np.array([-2,-1,0,1,2])
dc_bx = [-2,-1,0,1,2]
print(dc_bx)
print(ab)
# print(ab.shape)
# print(dc_bx.shape)
dc_by = []
dc_bd = []
weight_list = []
error_list = []
slop = -1 * w[0][0]/w[0][1]
# x = len(dc_bx)
# print(x)
for i in dc_bx:
    dc_by.append(slop * i + w[0][2]/w[0][1])

dc_bd.append(dc_by)
print(dc_bd)
# print(w)
epochs = 1000
print(f"epochs \t\t error")
for i in range(epochs+1):
    vk = np.dot(w, input)
    y = bipolar_sigmoid(vk)
    # print(y)
    e = target - y
    eta = .3
    delta = eta * np.dot(e, np.transpose(input))
    # print(delta)
    w = w + delta
    weight_list.append(w[0]);
    error_list.append(sse(e[0]/4))
    if i!=0 and i%1000 == 0:
        print(f"{i} \t\t {error_list[i]}")
        slop = -1 * weight_list[i][0]/weight_list[i][1]
        dc_by = []
        for x in dc_bx:
            dc_by.append(slop* x + weight_list[i][2]/weight_list[i][1])
        dc_bd.append(dc_by)

# print(np.round(y))

fig, ax = plt.subplots()
ax.plot([])



