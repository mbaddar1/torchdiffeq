import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.stats import norm, normaltest
from scipy.stats._mstats_basic import mquantiles
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from OT.geometric_quantile import obj_fn_1d

if __name__ == '__main__':
    N = 10000
    u = np.array(np.arange(0.1, 1, 0.01))
    x = norm.rvs(loc=0, scale=1, size=100).reshape(-1, 1)
    mio = 6
    sigma = 5.0
    alpha = 0.1
    y = norm.rvs(loc=mio, scale=sigma, size=100)
    reg = LinearRegression().fit(x, y)
    y_hat = reg.predict(x)
    mse = mean_squared_error(y_true=y, y_pred=y_hat)
    norm_test_naive = normaltest(y_hat)
    mean_y_hat_naive = np.nanmean(y_hat)
    std_y_hat_naive = np.std(y_hat)
    # Start q->q regression

    mean_ = np.nanmean(x)
    q0 = np.array([mean_] * len(u))
    min_ = minimize(fun=obj_fn_1d, x0=q0, args=(u, x), method='CG')
    qx = min_.x
    qxref = mquantiles(a=x.reshape(1, -1), prob=u)
    qx_err_norm = np.linalg.norm(qx - qxref)
    min_ = minimize(fun=obj_fn_1d, x0=q0, args=(u, y.reshape(-1, 1)), method='CG')
    qy = min_.x
    qyref = mquantiles(a=x.reshape(1, -1), prob=u)
    qy_err_norm = np.linalg.norm(qy - qyref)
    regq = LinearRegression().fit(qx.reshape(-1, 1), qy)
    yq_hat = regq.predict(qx.reshape(-1, 1))
    mseq = mean_squared_error(y_true=qy, y_pred=yq_hat)
    y_mapped = (sigma * x + mio)
    y_mapped_hat = regq.predict(x)
    mse_mapping = mean_squared_error(y_true=y_mapped, y_pred=y_mapped_hat)
    norm_test_qq = normaltest(y_mapped_hat)
    mean_y_hat_qq = np.nanmean(y_mapped_hat)
    std_y_hat_qq = np.std(y_mapped_hat)
    print(f"mse ofr x->y = {mse}")
    print(f"msq for q(x) -> q(y) Linear regression = {mse_mapping}")
    print(f'for y_hat_naive => normality-test = {norm_test_naive} , mean = {mean_y_hat_naive},sigma = {std_y_hat_naive}')
    print(f'for y_hat_naive => normality-test = {norm_test_qq} , mean = {mean_y_hat_qq},sigma = {std_y_hat_qq}')
    print("finished")
