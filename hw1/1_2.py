import random

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':


    red_x_list = []
    red_y_list = []
    blue_x_list = []
    blue_y_list = []

    for i in range (0, 1000):
        x = random.randrange(0, 100)
        y = random.randrange(0, 100)

        if x - 4 < y:
            red_x_list.append(x)
            red_y_list.append(y)
        else:
            blue_x_list.append(x)
            blue_y_list.append(y)

    print('red_x_list = ', red_x_list)
    #print('red_y_list =', red_y_list)
    #print('blue_x_list =', blue_x_list)
    #print('blue_y_list =', blue_y_list)


    x = np.linspace(0, 100, 100)
    y = 1.015 * x -4
    plt.plot(x, y, '-r', label='.')


    """..........................."""
    x = np.linspace(0, 100, 100)
    y = 1.01 * x - 4
    plt.plot(x, y, '-k', label='.')






    plt.scatter(red_x_list, red_y_list, marker=".", edgecolors='none')
    plt.scatter(blue_x_list, blue_y_list, marker=".", edgecolors='none')


    plt.xlabel('x_1')
    plt.ylabel('x_2')

    plt.show()
