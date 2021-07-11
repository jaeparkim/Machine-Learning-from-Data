import numpy as np
import random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
import sys
import math

if __name__ == '__main__':
    '''
    x = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]
    y = [0.9023, 0.3896, 0.0612, 0, 0, 0, 0]
    plt.axis([0.0, 1.0, 0, 1.0])

    # plt.hist(y, density=False, bins=7)  # density=False would make counts
    # plt.ylabel('Probability')
    plt.bar(x, y, width=1/6)

    plt.xlabel('ε')

    plt.show()

    sys.exit()
    '''

    # frequencies
    plt.axis([-0.5, 6, 0, 1])
    x = ['0', '1/6', '1/3', '1/2', '2/3', '5/6', '1']
    y = [0.9023, 0.3896, 0.0612, 0, 0, 0, 0]
    # setting the ranges and no. of intervals
    plt.bar(x,y)

    # x-axis label
    plt.xlabel('range of ε')
    # frequency label
    plt.ylabel('probability')
    # plot title

    # function to show the plot


    # setting the x - coordinates 
    x = np.arange(0, 1, 0.01)

    # setting the corresponding y - coordinates 
    y = 2 * np.exp(-12 * x**2)
    x = x * 6
    print(y)
    # potting the points 
    plt.plot(x, y, color='orange')

    # function to show the plot 
    plt.show()