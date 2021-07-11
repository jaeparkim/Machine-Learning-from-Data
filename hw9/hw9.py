import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy.linalg import inv
from random import shuffle


def preprocessing():
    one = []
    notOne = []
    with open('ZipDigits.train', 'r') as a, open('ZipDigits.test', 'r') as b:
        for line in a and b:
            temp = str.split(line)
            if temp[0] == '1.0000':
                one.append([float(i) for i in temp[1:]])
            else:
                notOne.append([float(i) for i in temp[1:]])

    return one, notOne


def intensity(digits):
    intensity = []
    for digit in digits:
        intensity.append(sum(digit) / len(digit))

    return intensity


def symmetric(digits):
    symmetry = []
    for digit in digits:
        arr = np.reshape(digit, (16, 16))
        horizontal_val = compare(arr, arr[::-1])
        vertical_val = compare(arr, arr[:, ::-1])
        symmetry.append(horizontal_val / 2 + vertical_val / 2)

    return symmetry


def compare(arr1, arr2):
    tmp = np.abs(arr1 - arr2)

    return np.sum(tmp) / 256


def normalizer(data):
    _min = min(data)
    _max = max(data)
    shift = (_max - _min) / 2 - _max
    scale = 1 / (_max + shift)

    return shift, scale


def normalization(data, shift, scale):
    data = [scale * (item + shift) for item in data]
    return data


def dataset_prep(x1_one, x2_one, x1_notone, x2_notone):
    test_set = []
    for i in range(0, len(x1_one)):
        test_set.append([x1_one[i], x2_one[i], 1])

    for i in range(0, len(x1_notone)):
        test_set.append([x1_notone[i], x2_notone[i], -1])
    shuffle(test_set)
    temp = np.random.randint(len(test_set), size=300)
    temp[::-1].sort()

    training_set = []
    for i in temp:
        training_set.append(test_set[i])
        del test_set[i]

    return training_set, test_set


# recursive Legendre
def L(x, k):
    if k == 0:
        return 1
    elif k == 1:
        return x
    else:
        return ((2*k-1)/k) * x * L(x, k-1) - ((k-1)/k) * x * L(x, k-2)


def L_transformation(data, order):
    temp = []
    for i in range(order + 1):
        for j in range(i + 1):
            temp.append(L(data[0], i - j) * L(data[1], j))

    temp.append(data[2])

    return temp


def f(x1, x2, n, w):
    result, idx = 0, 0
    for i in range(n + 1):
        for j in range(n - i + 1):
            result += w[idx] * L(x1, i) * L(x2, i)
            idx += 1

    return result


def linear_regression(dataSet, Lambda):
    Z = np.array([i[0:45] for i in dataSet])
    Y = np.array([i[45] for i in dataSet])
    Z_t = Z.T

    Z_t_Z = Z_t.dot(Z)
    Z_t_ZI = inv(Z_t_Z + Lambda * np.identity(Z_t_Z.shape[0]))
    w = Z_t_ZI.dot(Z_t).dot(Y)

    return w


def eighth_order_L(x1, x2, weight, order):
    result = 0
    index = 0
    for i in range(order + 1):
        for j in range(order - i + 1):
            result += weight[index] * L(x1, i) * L(x2, j)
            index += 1

    return result


def cross_validation(Lambda, trainSet, testData):
    w_reg = linear_regression(trainSet, Lambda)
    y_test = f(testData[1], testData[2], 8, w_reg)
    if testData[-1] == np.sign(y_test):
        return 0
    else:
        return (y_test - testData[-1]) ** 2


def approx_Etest(Lambda, train_set, test_set):
    w_reg = linear_regression(train_set, Lambda)
    E_out = 0
    for testData in test_set:
        y_test = f(testData[0], testData[1], 8, w_reg)
        if testData[2] == np.sign(y_test):
            continue
        else:
            E_out += (y_test - testData[2]) ** 2

    return E_out / len(test_set)


def plot(data, w):
    x1 = []
    y1 = []
    xNot1 = []
    yNot1 = []

    for i in range(len(data)):
        if data[i][2] == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            xNot1.append(data[i][0])
            yNot1.append(data[i][1])

    X, Y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    plt.contour(X, Y, eighth_order_L(X, Y, w, 8), levels=[0], colors='violet')
    plt.scatter(x1, y1, s=12, c='None', edgecolor='blue', marker='o')
    plt.scatter(xNot1, yNot1, s=12, c='red', marker='x')
    plt.xlabel('average intensity')
    plt.ylabel('symmetry')
    plt.show()


if __name__ == '__main__':
    one, not_one = preprocessing()

    intens_one = intensity(one)
    intens_not_one = intensity(not_one)
    symm_one = symmetric(one)
    symm_not_one = symmetric(not_one)

    # data normalization
    shift, scale = normalizer(intens_one + intens_not_one)
    intens_one = normalization(intens_one, shift, scale)
    intens_not_one = normalization(intens_not_one, shift, scale)

    shift, scale = normalizer(symm_one + symm_not_one)
    symm_one = normalization(symm_one, shift, scale)
    symm_not_one = normalization(symm_not_one, shift, scale)

    train_set, test_set = dataset_prep(intens_one, symm_one, intens_not_one, symm_not_one)

    # Problem 1
    train_trans = []
    for train in train_set:
        train_trans.append(L_transformation(train, order=8))

    # Problem 2
    w_reg_2 = linear_regression(train_trans, 0)
    plot(train_set, w_reg_2)

    # Problem 3
    w_reg_3 = linear_regression(train_trans, 2)
    plot(train_set, w_reg_3)

    # Problem 4
    E_cv_pts = []
    E_out_pts = []
    E_cv_min = 1000
    Lambda_star = 0

    domain = np.arange(0.1, 2.01, 0.1)
    for Lambda in domain:
        E_n = 0
        for j in range(0, 300):
            tmp = [item for item in train_trans]
            testData = tmp[j]
            del tmp[j]
            E_n += cross_validation(Lambda, tmp, testData)
        E_cv_pts.append(E_n / 300)
        if E_n / 300 < E_cv_min:
            E_cv_min = E_n / 300
            Lambda_star = Lambda
        E_out_pts.append(approx_Etest(Lambda, train_trans, test_set))

    plt.figure()
    plt.plot(domain, E_cv_pts, color='tomato', label='E_cv')
    plt.plot(domain, E_out_pts, color='violet', label='E_test')
    plt.legend()
    plt.show()

    # Problem 5
    print('lambda* : {}\nminimum E_cv: {}'.format(Lambda_star, E_cv_min))
    w_reg = linear_regression(train_trans, Lambda_star)
    plot(train_set, w_reg)

    sys.exit()
    
