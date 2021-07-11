import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import sys


def euclidean(v1, v2):
    return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5


def transform(x):
    tmp = []
    for i in range(len(x)):
        if x[i][0] == 0 and x[i][1] == 0:
            z = np.arctan(0)
        elif x[i][0] == 0:
            z = np.sign(x[i][1]) * np.arctan(math.inf)
        else:
            z = np.arctan(x[i][1] / x[i][0])

        tmp.append([math.sqrt(x[i][0] ** 2 + x[i][1] ** 2), z])

    return tmp


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


if __name__ == '__main__':

    option = sys.argv[1]
    k = int(sys.argv[2])

    x1True = [0.0, 0.0, -2.0]
    x2True = [2.0, -2.0, 0.0]
    x1False = [1.0, 0.0, 0.0, -1.0]
    x2False = [0.0, 1.0, -1.0, 0.0]

    x = list(zip(x1True, x2True))
    y = [1 for i in range(len(x1True))]
    x = x + list(zip(x1False, x2False))
    y = y + [-1 for i in range(len(x1False))]

    x1_T, x2_T, x1_F, x2_F = [], [], [], []
    if option == 'b':
        x = transform(x)

    for i in np.arange(-4, 4, 0.02):
        for j in np.arange(-4, 4, 0.02):
            if option == 'a':

                if knn([i, j], x, y, k):
                    x1_T.append(i)
                    x2_T.append(j)
                else:
                    x1_F.append(i)
                    x2_F.append(j)

            else:
                if i == 0 and j == 0:
                    z = np.arctan(0)
                elif i == 0:
                    z = np.sign(j) * np.arctan(math.inf)
                else:
                    z = np.arctan(j / i)

                if knn([math.sqrt(i ** 2 + j ** 2), z], x, y, k):
                    x1_T.append(i)
                    x2_T.append(j)
                else:
                    x1_F.append(i)
                    x2_F.append(j)

    plt.scatter(x1_T, x2_T, s=10, color='violet', label='true: +1')
    plt.scatter(x1_F, x2_F, s=10, color='turquoise', label='false: -1')
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
