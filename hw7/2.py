import matplotlib.pyplot as plt
import sys
import math as m

def gradient_descent(x, y, eta):

    f_list = []

    for i in range(0, 50):
        f = x ** 2 + y ** 2 + 2 * m.sin(2 * m.pi * x) * m.sin(2 * m.pi * y)
        x_new = x - eta * (2 * x + 4 * m.pi * m.cos(2 * m.pi * x) * m.sin(2 * m.pi * y))
        y_new = y - eta * (4 * y + 4 * m.pi
                           * m.sin(2 * m.pi * x) * m.cos(2 * m.pi * y))
        x = x_new
        y = y_new
        f_list.append(f)
        
    print('x: {}\ny: {}'.format(x, y))

    print('f final: %f \n' % f_list[49])

    plt.scatter(list(range(1, 51)), f_list, s=8, color='violet')
    # plt.plot(list(range(1, 51)), f_list, color='violet') # no convergence

    plt.xlabel('number of iterations')
    plt.ylabel('f')
    plt.show()


if __name__ == '__main__':

    gradient_descent(0.1, 0.1, 0.01)
    gradient_descent(0.1, 0.1, 0.1)
    
    '''
    gradient_descent(0.1, 0.1, 0.01)
    gradient_descent(1, 1, 0.01)
    gradient_descent(-0.5, -0.5, 0.01)
    gradient_descent(-1, -1, 0.01)
    print('-----------------------------------------')
    gradient_descent(0.1, 0.1, 0.1)
    gradient_descent(1, 1, 0.1)
    gradient_descent(-0.5, -0.5, 0.1)
    gradient_descent(-1, -1, 0.1)
    '''

    sys.exit()
