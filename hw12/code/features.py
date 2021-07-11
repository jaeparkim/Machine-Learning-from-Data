import numpy as np


def symmetry(array):
    sum_1 = 0
    for i in range(0, 8):
        for j in range(0, 16):
            sum_1 += abs(float(array[j * 16 + i]) - float(array[(j + 1) * 16 - (i + 1)]))

    sum_2 = 0
    for i in range(0, 16):
        for j in range(0, 8):
            sum_2 += abs(float(array[j * 16 + i]) - float(array[(15 - j) * 16 - i]))

    return (sum_2 + sum_1) / 128


def intensity(array):
    return np.sum(np.absolute([float(i) for i in array]))
