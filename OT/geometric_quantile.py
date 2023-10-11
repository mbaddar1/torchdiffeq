"""

Implement Geometric Quantiles
original paper
CHAUDHURI 96
http://dspace.isical.ac.in:8080/jspui/bitstream/10263/6795/1/On%20a%20Geometric%20Notion%20of%20Quantiles%20for%20Multivariate%20Data.pdf
my copy
https://drive.google.com/file/d/1uUFY3_sV7Muzgrr81IPN-YUwWTJKPtnI/view?usp=drive_link


FROM GEOMETRIC QUANTILES TO HALFSPACE DEPTHS: A GEOMETRIC
APPROACH FOR EXTREMAL BEHAVIOUR.
https://essec.hal.science/hal-04134321/document
section 2.1 p 5

Extreme geometric quantiles (Lecture)
https://mistis.inrialpes.fr/~girard/Fichiers/slides_ERCIM_2014.pdf

Statistical properties of approximate
geometric quantiles in
infinite-dimensional Banach spaces
https://arxiv.org/pdf/2211.00035.pdf
"""
import numpy as np
import scipy.stats
import torch
from scipy.optimize import minimize
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy.stats import norm


def obj_fn_1d(q: np.array, u: float, X: np.ndarray):
    # assert q.ndim == 1
    # assert u.ndim == 1
    # assert q.shape[0] == u.shape[0]
    # assert X.ndim == 2
    # assert X.shape[1] == q.shape[0]
    # N = X.shape[0]
    # dump1 = X - q
    # dump2 = np.linalg.norm(X)

    X_list = X.tolist()

    # def myfunc(x):
    #     return x - q

    A = list(map(lambda x: np.abs(np.array(x) - q) - np.abs(x), X_list))
    E_A = np.mean(A, axis=0)
    # term2 = np.array(list(map(lambda g:np.linalg.norm(g),X_list))).reshape(-1,1)
    # term1 = 1.0 / N * np.sum(np.linalg.norm(X - q) - term2)
    # sum_term = np.
    B = (2 * u - 1) * q  # np.dot(u, q)

    obj = E_A - B
    # print(f'objective = {np.sum(obj)}')
    return np.sum(obj)


if __name__ == '__main__':
    # np.random.sample(size=1)[0]
    u = np.array(np.arange(0.0, 1, 0.05))
    a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    mean_ = np.nanmean(a)
    q0 = np.array([mean_] * len(u))
    mq = mquantiles(a)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    # X = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.]).reshape(-1, 1)
    # X = np.array([7, 15, 36, 39, 40, 41]).reshape(-1, 1)
    X = norm.rvs(size=200).reshape(-1, 1)
    q_test = norm.ppf(u)
    print(f'qtest_closed_form = {q_test}')
    q_test_mquantiles_scipy = mquantiles(a=X, prob=u)
    print(f'q_test_mquantiles_scipy = {q_test_mquantiles_scipy}')
    min_ = minimize(fun=obj_fn_1d, x0=q0, args=(u, X), method='CG') # CG proved to be good
    print(min_)
    # for q in np.arange(10.0, 50, 0.1):
    #     print(f'q = {q} => obj-fn = {obj_fn(q=[q], u=u, X=X)}')
