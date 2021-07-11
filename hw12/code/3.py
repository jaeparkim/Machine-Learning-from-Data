import numpy as np
import matplotlib.pyplot as plt


t = np.arange(-1.6, 1.6, 0.1)
f, = plt.plot([0] * len(t), t, 'r')
g, = plt.plot(t, t ** 3, 'b')
plt.legend([f, g], ["X-space decision boundary", "Z-space decision boundary"])
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
plt.title("Decision boundary")
plt.show()
