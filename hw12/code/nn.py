import numpy as np
import sys


class nn:
    def __init__(self, _input=2, hidden=10, output=1):
        self.input_dim = _input
        self.hidden_dim = hidden
        self.output_dim = output
        np.random.seed(0)

        self.w1 = np.random.randn(_input + 1, hidden)
        self.w2 = np.random.randn(hidden + 1, output)

    def forward(self, x):
        self.x0 = np.insert(x, 0, 1, axis=1)
        self.s0 = self.x0.dot(self.w1)
        self._x1 = np.tanh(self.s0)
        self.x1 = np.insert(self._x1, 0, 1, axis=1)
        self._s2 = self.x1.dot(self.w2)
        # self.s2 = np.tanh(self._s2)

    def backprop(self, y):
        self.d2 = 2 * (self._s2 - y)
        self.dw2 = self.x1.T.dot(self.d2)
        self.d1 = self.d2.dot(self.w2[1:].T) * (1 - self._x1 ** 2)
        self.dw1 = self.x0.T.dot(self.d1)

    def train(self, x, y, epsilon=0.0001):
        self.forward(x)
        self.backprop(y)
        self.w1 -= epsilon * self.dw1
        self.w2 -= epsilon * self.dw2

    def predict(self, x):
        self.forward(x)
        return np.sign(self._s2)

    def error(self, x, y):
        self.forward(x)
        error = np.linalg.norm((np.sign(self._s2) - y), ord=2)
        return error / len(y)


if __name__ == '__main__':
    sys.exit()
