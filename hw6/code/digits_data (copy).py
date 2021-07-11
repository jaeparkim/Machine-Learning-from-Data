import numpy as np
import matplotlib.pyplot as plt
import sys


def intensity(image):
    intensity_sum = sum(image)
    avg_intensity = intensity_sum / 256

    return avg_intensity


def symmetric(data):
    # reshape 1D data to 2D image of pixels 16 x 16
    image = np.reshape(data, (16, 16))

    # compare intensity of two pixels across from each other
    # along the horizontal axis of the image
    temp_img = np.zeros((8, 16))
    temp_img = image[0:8][0:16] - image[-1:-9:-1][0:16]
    temp_img = np.absolute(temp_img)

    normalized_sum = np.sum(temp_img) / 256

    return normalized_sum


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python", sys.argv[0], "input_file_name.ext")
        sys.exit(0)

    # dictionary to hold pixel data with corresponding digits
    digits = {0: [], 1: [], 2: [], 3: [], 4: [],
              5: [], 6: [], 7: [], 8: [], 9: []}

    with open(sys.argv[1], 'r') as file:
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

    intensity_vals = []
    symmetry_vals = []

    # for digit 1
    for data in digits[1]:
        intensity_vals.append(intensity(data))
        symmetry_vals.append(symmetric(data))

    plt.scatter(intensity_vals, symmetry_vals, color='none', edgecolor='b', s=10)

    intensity_vals.clear()
    symmetry_vals.clear()

    # for digit 5
    for data in digits[5]:
        intensity_vals.append(intensity(data))
        symmetry_vals.append(symmetric(data))

    plt.plot(intensity_vals, symmetry_vals, "rx", markersize=5)

    plt.xlabel("average intensity")
    plt.ylabel("symmetric")
    plt.show()

    sys.exit()
