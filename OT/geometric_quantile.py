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
from torch import vmap
from torch.nn import MSELoss
import random

#
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def _get_diffs(x: torch.Tensor, q: torch.Tensor):
    term = torch.norm(x - q) - torch.norm(x)
    return term


def geometric_mean_obj_fn_ND(q: np.array, u: torch.Tensor, X: torch.Tensor):
    q_tensor = torch.tensor(q)
    diffs = vmap(lambda x: _get_diffs(x, q_tensor))(X)
    exp_diffs = torch.mean(diffs).item()
    u_dot_q = torch.dot(u, q_tensor.type(torch.float32)).item()
    obj = exp_diffs - u_dot_q
    # print(obj)
    return obj


def geometric_mean_obj_fn_1D(q: np.array, u: float, X: np.ndarray):
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
    test_1d = False
    test_nd = True
    if test_1d:
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
        min_ = minimize(fun=geometric_mean_obj_fn_1D, x0=q0, args=(u, X), method='CG')  # CG proved to be good
        print(min_)
    if test_nd:
        N = 20000
        # test vmap
        t = torch.tensor([1, 2, 3, 4]).view(-1, 1)
        r = vmap(lambda x: torch.pow(x, 2))(t)
        # test get_diffs
        q_ = torch.tensor([0.1, 0.2])
        x_ = torch.tensor([[0.4, 0.5], [0.6, 0.7]])

        k = vmap(lambda j: _get_diffs(j, q_))(x_)
        assert MSELoss()(k, torch.tensor([-0.216, -0.21485])) <= 1e-4
        # start the real work
        D = 3
        opt_method = 'Nelder-Mead'
        X_base = (torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=torch.eye(D))
                  .sample(torch.Size([N])))
        u = torch.tensor([0.2, 0.8, 0.3])
        q0 = torch.mean(X_base, dim=0).detach().numpy()
        min_ = minimize(fun=geometric_mean_obj_fn_ND, x0=q0, args=(u, X_base),
                        method=opt_method)  # CG proved to be good
        print(min_)
        q_base = min_.x
        mio = torch.tensor([2.0, 1.0, 7.0])
        Sigma = torch.diag(torch.tensor([0.25, 0.36, 0.09]))
        # A = torch.sqrt(Sigma)
        A = torch.diag(torch.tensor([0.5, 0.6, 0.3]))
        X_target = torch.distributions.MultivariateNormal(loc=mio, covariance_matrix=Sigma).sample(torch.Size([N]))
        X_target2 = torch.einsum('ij,bj->bi', A, X_base) + mio
        min_ = minimize(fun=geometric_mean_obj_fn_ND, x0=q0, args=(u, X_target2),
                        method=opt_method)  # CG proved to be good
        q_target_gq = min_.x
        q_target_expected = torch.matmul(A, torch.tensor(q_base).type(torch.float32)) + mio
        mse_loss = MSELoss()(q_target_expected, torch.tensor(q_target_gq))
        print(min_)
        print(f'mse_loss = {mse_loss}')
        assert mse_loss < 0.05, f"mse = {mse_loss}"
        print("finished")
