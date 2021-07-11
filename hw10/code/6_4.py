import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import sys


def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5


def knn(test, xz, y, k):
    true = 0
    false = 0
    distances = []
    for train in xz:
        distances.append(euclidean(test, train))
    points = map(distances.index, heapq.nsmallest(k, distances))
    for i in points:
        if y[i] == 1:
            true += 1
        else:
            false += 1

    return true > false


def random_points(n=1000):
    r1 = 15
    r2 = 10
    x_1 = 0
    y_1 = 0
    x_2 = 12
    y_2 = -5
    count = 1

    one = []
    notone = []
    while count <= n:
        x = np.random.uniform(-r1, r1)
        y = np.random.uniform(-r1, r1)
        d = x ** 2 + y ** 2
        if r2 ** 2 <= d <= r1 ** 2:
            if y > 0:
                one.append([x_1 + x, y_1 + y])
            else:
                notone.append([x_2 + x, y_2 + y])
            count += 1
        else:
            continue

    return one, notone


if __name__ == '__main__':
    k = int(sys.argv[1])

    one, notone = random_points(1000)

    x1True = [float(i[0]) for i in one]
    x2True = [float(i[1]) for i in one]
    x1False = [float(i[0]) for i in notone]
    x2False = [float(i[1]) for i in notone]

    x = list(zip(x1True, x2True))
    y = [1 for i in range(len(x1True))]
    x = x + list(zip(x1False, x2False))
    y = y + [-1 for i in range(len(x1False))]

    x1_T, x2_T, x1_F, x2_F = [], [], [], []

    for i in np.arange(-23, 30, 0.2):
        for j in np.arange(-25, 20, 0.2):
            if knn([i, j], x, y, k):
                x1_T.append(i)
                x2_T.append(j)
            else:
                x1_F.append(i)
                x2_F.append(j)


    plt.scatter(x1_T, x2_T, s=10, color='violet', label='true: +1')
    plt.scatter(x1_F, x2_F, s=10, color='turquoise', label='false: -1')
    plt.scatter(x1True, x2True, s=7, marker='o', color='red')
    plt.scatter(x1False, x2False, s=7, marker='o', color='blue')
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
