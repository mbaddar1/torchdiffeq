"""
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

if __name__ == '__main__':
    x = np.arange(10)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig('orig.jpg')
    plt.clf()
    cs = CubicSpline(x, y)
    xs = np.arange(-0.5, 9.6, 0.1)
    ys = cs(xs)
    plt.plot(xs, ys)
    plt.savefig('cs.jpg')
    plt.show()
