from random import random

import numpy as np
from scipy.stats import norm, normaltest

if __name__ == '__main__':
    u = np.random.uniform(low=0, high=1.0, size=1000000)
    samples_ = norm.ppf(q=u, loc=-8, scale=5)
    res = normaltest(a=samples_)
    print(f'mean = {np.mean(samples_)}')
    print(f'scale = {np.std(samples_)}')
    print(f" p = {res.pvalue}")
