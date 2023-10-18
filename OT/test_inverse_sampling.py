from random import random

import numpy as np
import torch
from scipy.stats import norm, normaltest
import pingouin as pg
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # u = np.random.uniform(low=0, high=1.0, size=1000000)
    # samples_ = norm.ppf(q=u, loc=-8, scale=5)
    # res = normaltest(a=samples_)
    # print(f'mean = {np.mean(samples_)}')
    # print(f'scale = {np.std(samples_)}')
    # print(f" p = {res.pvalue}")
    D = 4
    N = 10000
    u = np.random.uniform(size=(D, N))
    loc = np.array([0, 2, -8, 10])
    Scale = np.array([1, 5, 0.2, 8])
    u_list = list(u)
    sample = np.transpose(np.array([norm.ppf(q=u_list[i], loc=loc[i], scale=Scale[i]) for i in range(D)]))
    sample_mean = np.mean(sample, axis=0)
    sample_cov = np.cov(np.transpose(sample))
    norm_test_res = pg.multivariate_normality(X=sample)
    print(f'sample_mean mse = {mean_squared_error(sample_mean, loc)}')
    cov_ref = np.diag(Scale) ** 2
    print(f'Test Inverse Sampling')
    print(f'sample_scale mse = {mean_squared_error(sample_cov, cov_ref)}')
    print(f'norm-test-res = {norm_test_res}')
    print('---')
    sample2 = np.random.multivariate_normal(mean=loc, cov=cov_ref,size=N)
    print('Test direct sampling as a benchmark')
    print(f'sample mean mse = {mean_squared_error(np.mean(sample2, axis=0), loc)}')
    sample_cov = np.cov(np.transpose(sample2))
    print(f'sample cov mse = {mean_squared_error(sample_cov,cov_ref)}')
    print(f'norm-test-direct = {pg.multivariate_normality(sample2)}')
    print('finished')
