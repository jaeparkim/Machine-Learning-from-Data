import numpy as np
import matplotlib.pyplot as plt
import sys
from nn import nn
from features import symmetry, intensity
from random import shuffle


if __name__ == "__main__":
    option = sys.argv[1]
    num_iters = 50  # 2000000

    x_one, y_one, x_notone, y_notone = [], [], [], []
    x_s, y_s, X, Y = [], [], [], []
    data = []

    f = open('ZipDigits.train', 'r')
    for line in f:
        line = line.split(' ')
        x = intensity(line[1:-1])
        y = symmetry(line[1:-1])
        x_s.append(x)
        y_s.append(y)
        if line[0] == '1.0000':
            y_one.append(y)
            x_one.append(x)
        else:
            y_notone.append(y)
            x_notone.append(x)
    f.close()

    f = open('ZipDigits.test', 'r')
    for line in f:
        line = line.split(' ')
        x = intensity(line[1:-1])
        y = symmetry(line[1:-1])
        x_s.append(x)
        y_s.append(y)
    f.close()

    x_s = np.array(x_s)
    y_s = np.array(y_s)
    x_one = np.array(x_one)
    x_one = 2 * (x_one - x_s.min()) / x_s.ptp() - 1
    y_one = np.array(y_one)
    y_one = 2 * (y_one - y_s.min()) / y_s.ptp() - 1
    x_notone = np.array(x_notone)
    x_notone = 2 * (x_notone - x_s.min()) / x_s.ptp() - 1
    y_notone = np.array(y_notone)
    y_notone = 2 * (y_notone - y_s.min()) / y_s.ptp() - 1
    plt.plot(x_one, y_one, 'bo', markersize=1, label="Digit 1")
    plt.plot(x_notone, y_notone, 'ro', markersize=1, label="Others")

    # load
    for i in range(len(x_one)):
        if option == 'c':
            ptx = [x_one[i], y_one[i], 1]
            data.append(ptx)
        else:
            ptx = [x_one[i], y_one[i]]
            Y.append([1])
            X.append(ptx)
    for i in range(len(y_notone)):
        if option == 'c':
            ptx = [x_notone[i], y_notone[i], -1]
            data.append(ptx)
        else:
            ptx = [x_notone[i], y_notone[i]]
            Y.append([-1])
            X.append(ptx)

    # preprocessing
    if option == 'a' or option == 'b':
        X = np.array(X)
        Y = np.array(Y)
        error = []
    else:
        shuffle(data)
        traind = data[:250]
        testd = data[250:]
        for d in traind:
            X.append([d[0], d[1]])
            Y.append([d[2]])

        X = np.array(X)
        Y = np.array(Y)

        Xt = []
        Yt = []
        for d in testd:
            Xt.append([d[0], d[1]])
            Yt.append([d[2]])
        ein = []
        ecv = []


    iteration = []
    model = nn()

    for i in range(num_iters):
        iteration.append(i + 1)
        epsilon = 0.0004 / np.sqrt(i + 1)
        if option == 'a':
            model.train(X, Y)
        else:
            model.train(X, Y, epsilon)

        e1 = model.error(X, Y)
        if option == 'a' or option == 'b':
            error.append(e1)
        else:
            ein.append(e1)
            e1 = model.error(Xt, Yt)
            ecv.append(e1)


    xt = []
    yt = []
    data_test = []
    f = open('ZipDigits.test', 'r')
    for line in f:
        line = line.split(' ')
        y = symmetry(line[1:-1])
        x = intensity(line[1:-1])
        if line[0] == '1.0000':
            data_test.append((x, y, 1))
        else:
            data_test.append((x, y, -1))
    f.close()
    for i in data_test:
        x = 2 * (i[0] - x_s.min()) / x_s.ptp() - 1
        y = 2 * (i[1] - y_s.min()) / y_s.ptp() - 1
        ptx = [x, y]
        xt.append(ptx)
        label = i[2]
        yt.append([label])
    Xt = np.array(xt)
    Yt = np.array(yt)
    e1 = model.error(Xt, Yt)
    print("test_error: ", e1)


    X, Y = np.meshgrid(np.arange(-1, 1, 0.001),
                       np.arange(-1, 1, 0.001))
    x_len, y_len = np.shape(X)
    tptx = []
    for i in range(x_len):
        for j in range(y_len):
            px = X[i][j]
            py = Y[i][j]
            ptx = [px, py]
            tptx.append(ptx)

    z = model.predict(np.array(tptx))
    Z = np.reshape(z, (x_len, y_len))
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.contourf(X, Y, Z, colors=('gray', 'white'))
    plt.title("Decision boundary")
    plt.legend(loc='upper right')
    plt.show()


    if option == 'a' or option == 'b':
        f, = plt.plot(iteration, error, 'ko', markersize=2)
        plt.legend([f], [r"$E_{in}$"])
    elif option == 'c':
        f, = plt.plot(iteration, ein, 'ro', markersize=2)
        g, = plt.plot(iteration, ecv, 'bo', markersize=2)
        plt.legend([f, g], [r"$E_{in}$", r"$E_{validation}$"])
    plt.xlabel("$iterations$")
    plt.ylabel('Error')
    plt.title("In sample error")
    plt.show()

