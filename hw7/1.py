import numpy as np
import math
from numpy.linalg import inv
from matplotlib import pyplot as plt
import sys


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


def load_values(img, one, five):
    for line in img:
        if line[0] == 1 or line[0] == 5:
            data = line[1:]
            arr = np.reshape(data, (16, 16))

            intensity_val = sum(data) / len(data)
            symmetry_val = compare(arr, arr[::-1]) / 2 + (compare(arr, arr[:, ::-1])) / 2

            if line[0] == 1:
                one.append([symmetry_val, intensity_val])
            elif line[0] == 5:
                five.append([symmetry_val, intensity_val])


def compare(arr1, arr2):
    tmp = np.abs(arr1 - arr2)

    return np.sum(tmp) / 256


def preprocessing(file, option):
    if option == 'normal':
        with open(sys.argv[2], 'r') as file:
            if file is None:
                print("Failed to open", sys.argv[1])
                sys.exit(0)

            for line in file:
                line = file.readline()
                if line == "":
                    break
                data = np.array(str.split(line))
                digit = int(float(data[0]))
                temp = []
                for pixel in data[1:]:
                    temp.append(float(pixel))
                digits[digit].append(temp)

        intensity1_list = intensity(digits[1])
        intensity5_list = intensity(digits[5])
        symmetry1_list = symmetric(digits[1])
        symmetry5_list = symmetric(digits[5])
        print('Number of dataset: ' + str(len(digits[1]) + len(digits[5])))

        for i in range(0, len(intensity1_list)):
            one.append([intensity1_list[i], symmetry1_list[i]])

        for i in range(0, len(intensity5_list)):
            five.append([intensity5_list[i], symmetry5_list[i]])



    elif option == 'transform':
        load_values(np.loadtxt(sys.argv[2]), one, five)
        load_values(np.loadtxt(sys.argv[3]), one2, five2)


def algorithm(option):
    if option == 'normal':
        intensity1_list = intensity(digits[1])
        intensity5_list = intensity(digits[5])
        symmetry1_list = symmetric(digits[1])
        symmetry5_list = symmetric(digits[5])

        for i in range(0, len(intensity1_list)):
            one.append([intensity1_list[i], symmetry1_list[i]])

        for i in range(0, len(intensity5_list)):
            five.append([intensity5_list[i], symmetry5_list[i]])

        X_mat = np.array([[1] + i for i in one] + [[1] + i for i in five])
        Y_mat = np.array([1] * len(one) + [-1] * len(five))
        # one step algo
        w_lin = inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y_mat)
        print('W_lin of regression: [{}, {}, {}]'.format(w_lin[0], w_lin[1], w_lin[2]))

        x1 = [[1] + i + [1] for i in one]
        x2 = [[1] + i + [-1] for i in five]
        data = x1 + x2
        data = np.array(data)
        np.random.shuffle(data)

        w, error = PLA(data, 1, 1000, w_lin)

        print(
            'Error: {}\nW: {}, {}, {}\nE_in: {}'.format(int(error[-1] / 2), w[0], w[1], w[2], error[-1] / len(data)))

        plt.plot([-1.2, 0.5], [-(w[0] + w[1] * i) / w[2] for i in [-1, 0]], color='black')
        plt.scatter(intensity1_list, symmetry1_list, s=10, color='none', edgecolor='b')
        plt.scatter(intensity5_list, symmetry5_list, s=10, color='red', marker='x')
        plt.xlabel('average intensity')
        plt.ylabel('symmetry')
        plt.show()


    elif option == 'transform':
        X1 = [i[0] for i in one]
        Y1 = [i[1] for i in one]

        X2 = [i[0] for i in five]
        Y2 = [i[1] for i in five]

        plt.scatter([i[0] for i in one], [i[1] for i in one],  s=10, color='none', edgecolor='b')
        plt.scatter([i[0] for i in five], [i[1] for i in five],  s=10, color='red', marker='x')

        data = [[1] + i + [1] for i in one] + [[1] + i + [-1] for i in five]
        np.random.shuffle(np.array(data))

        # transformation
        result = []
        for i in data:
            x1 = i[1]
            x2 = i[2]
            flag = i[-1]
            x = [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2, x1 ** 3, (x1 ** 2) * x2, x1 * (x2 ** 2), x2 ** 3, flag]
            result.append(x)
        data_new = np.array(result)

        X_new = data_new[:, :-1]
        Y_new = data_new[:, -1]

        # third-order polynomial transformation
        X, Y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000))
        w = inv(X_new.T.dot(X_new)).dot(X_new.T).dot(Y_new)
        A = w[0] + w[1] * X + w[2] * Y + w[3] * X * Y + w[4] * (X ** 2) + w[5] * (Y ** 2) + w[6] * (X ** 3) + w[7] * (
                X ** 2) * Y + w[8] * X * (Y ** 2) + w[9] * Y ** 3

        plt.contour(X, Y, A, colors='gray')
        plt.scatter(X1, Y1, s=10, color='none', edgecolor='b')
        plt.scatter(X2, Y2, s=10, color='red', marker='x')
        plt.axis([-0.1, 1, -1.5, 0.5])
        plt.xlabel("Intensity")
        plt.ylabel("Symmetry")
        plt.show()

        error = num_error(data_new, w)
        print("Error: " + str(error))
        print('E_in: ' + str(float(num_error(data_new, w)) / float(data_new.shape[0])))

        X1 = [i[0] for i in one2]
        Y1 = [i[1] for i in one2]
        X2 = [i[0] for i in five2]
        Y2 = [i[1] for i in five2]
        plt.contour(X, Y, A, colors='gray')
        plt.scatter(X1, Y1,  s=10, color='none', edgecolor='b')
        plt.scatter(X2, Y2,  s=10, color='red', marker='x')
        plt.axis([-0.1, 1, -1.5, 0.5])
        plt.xlabel("Intensity")
        plt.ylabel("Symmetry")
        plt.show()


def PLA(x, k, max_num, w_lin):
    w = w_lin
    w_0 = w_lin
    # w = np.zeros(n)
    # w0 = np.zeros(n)
    m, n = x.shape
    n -= 1
    error = num_error(x, w)
    Error = []

    if error == 0:
        pass
    else:
        j = 0
        while j < max_num or error == 0:
            k = np.random.randint(0, m)
            i = x[k]
            w = w_0 + k * i[-1] * i[:n]
            error_1 = num_error(x, w)
            if error > error_1:
                w_0 = w[:]
                error = error_1
            Error.append(error)
            j += 1

    return w_0, Error


def num_error(data, w):
    n = data.shape[1] - 1
    count = 0
    for i in data:
        if np.sign(np.inner(i[:n], w)) * i[-1] < 0:
            count += 1

    return count


'''
python 1.py normal ZipDigits.train
python 1.py transform ZipDigits.train ZipDigits.test
'''
if __name__ == '__main__':
    digits = {0: [], 1: [], 2: [], 3: [], 4: [],
              5: [], 6: [], 7: [], 8: [], 9: []}

    option = sys.argv[1]  # normal or transform

    intensity1_list = []
    intensity5_list = []
    symmetry1_list = []
    symmetry5_list = []

    one = []
    five = []

    one2 = []
    five2 = []

    preprocessing(sys.argv[2], option)
    algorithm(option)
    sys.exit()
