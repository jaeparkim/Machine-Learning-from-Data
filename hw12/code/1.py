import numpy as np
import copy
import sys

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(np.negative(x)))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


def identity(x):
    return x


def identity_deriv(x):
    return 1


class NeuralNetwork:
    def __init__(self, network, hidden_activation, output_activation, init_weight=0.25):
        # 0 ~ N-1 layers
        if hidden_activation == 'tanh':
            self.hidden_activation = tanh
            self.hidden_dv = tanh_deriv
        # elif

        # output layer  
        if output_activation == 'tanh':
            self.output_activation = tanh
            self.output_dv = tanh_deriv
        elif output_activation == 'identity':
            self.output_activation = identity
            self.output_dv = identity_deriv
        elif output_activation == 'sign':
            self.output_activation = np.sign
            self.output_dv = 0

        # network layers
        self.layers = []
        for i in range(len(network)):
            layer = np.zeros(network[i])
            self.layers.append(layer)

        # weight edgess
        self.weights = []
        self.biases = []
        for i in range(len(network) - 1):
            w = np.full((network[i], network[i + 1]), init_weight)
            self.weights.append(w)
            b = np.full(network[i + 1], init_weight)
            self.biases.append(b)

        self.output = 0


    def forward(self, data):
        x, y = data

        for i in range(len(self.layers[0])):
            self.layers[0][i] = x[i]

        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]

            layer = self.layers[i]
            
            output_layer = np.matmul(layer, w) + b
            if i == len(self.weights) - 1:
                output_layer = self.output_activation(output_layer)
            else:
                output_layer = self.hidden_activation(output_layer)
            self.layers[i + 1] = output_layer

        answer = copy.deepcopy(self.layers[len(self.layers) - 1])
        print(self.layers)
        print(answer, '\n')

        return answer

    def squared_err(self, h, y):
        _y = h[0]
        return 0.25 * (_y - y) ** 2


    def gradient_prop(self, data, rate=0.0001):
        x, y = data

        out = self.forward(data)
        err = self.squared_err(out, y)

        network_grads = []
        for i in range(len(self.weights)):
            gradient_matrix = []
            for j in range(len(self.weights[i])):
                row_grads = []
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += rate

                    h = self.forward(data)
                    err_ = self.squared_err(h, y)
                    row_grads.append(float(err_ - err) / rate)

                    self.weights[i][j][k] -= rate
 
                gradient_matrix.append(row_grads)

            network_grads.append(gradient_matrix)

        for i in network_grads:
            print(i)
        
        bias_grads = []
        for i in range(len(self.biases)):
            bias_row_grads = []
            for j in range(len(self.biases[i])):
                self.biases[i][j] += rate

                h = self.forward(data)
                err_ = self.squared_err(h, y)
                bias_row_grads.append(float(err_ - err) / rate)

                self.biases[i][j] -= rate

            bias_grads.append(bias_row_grads)

        print()
        for j in bias_grads:
            print(j)



if __name__ == '__main__':
    
    m = 2
    hidden_activation = 'tanh'
    output_activation = 'tanh'

    network_shape = [2, m, 1]

    network = NeuralNetwork(network_shape, hidden_activation, output_activation, init_weight=0.25)
    
    data = [1, 2], 1

    network.forward(data)
    print(network.layers)

    network.gradient_prop(data, 0.0001)

    sys.exit()
