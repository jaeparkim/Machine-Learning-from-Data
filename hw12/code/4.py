import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from features import symmetry, intensity


if __name__ == "__main__":
    option = sys.argv[1]
    C_val = float(sys.argv[2])


    x_one, y_one, x_notone, y_notone = [], [], [], []
    x_s, y_s, X, Y = [], [], [], []

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
    plt.plot(x_one, y_one, 'bo', markersize='2', label="Digit 1")
    plt.plot(x_notone, y_notone, 'ro', markersize='2', label="Others")


    # load
    X = []
    Y = []
    for i in range(len(x_one)):
        ptx = [x_one[i], y_one[i]]
        Y.append(1)
        X.append(ptx)
    for i in range(len(y_notone)):
        ptx = [x_notone[i], y_notone[i]]
        Y.append(-1)
        X.append(ptx)
    X = np.array(X)
    Y = np.array(Y)


    if option == 'a':
        model = svm.SVC(kernel='poly', degree=8, coef0=1, C=C_val)
        model.fit(X, Y)
        py = model.predict(X)
        error = len(Y) - np.sum(py == Y)
        print("Ein: ", float(error) / len(Y))

    elif option == 'c':
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
            yt.append(label)
        Xt = np.array(xt)
        Yt = np.array(yt)
        model = svm.SVC(kernel='poly', degree=8, coef0=1, C=C_val)
        model.fit(X, Y)
        py = model.predict(Xt)
        error = len(Yt) - np.sum(py == Yt)
        print("Etest: ", float(error) / len(Yt))


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
    Z = model.predict(tptx)
    Z = Z.reshape(X.shape)
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.contourf(X, Y, Z, colors=('w', 'gray'))
    plt.title("Decision boundary for $C = {}$".format(C_val))
    plt.legend(loc='upper right')
    plt.show()

