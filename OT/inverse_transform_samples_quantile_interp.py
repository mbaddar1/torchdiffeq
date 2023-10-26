"""
https://stackoverflow.com/a/44163082

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

plan
1. Implement a function that takes quantiles and generate samples Done
2. Test Generated samples using WD distance
3. Test on Normal, Exp, LogNormal and Empirical Dist from ICA transformation
4- Use Cubic Spline
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from sklearn.decomposition import FastICA
from torch.nn import MSELoss
import pingouin as pg

from OT.utils import wasserstein_distance_two_gaussians


def uv_sample(Yq: torch.Tensor, N: int, u_levels: torch.Tensor, u_step: float, interp: str) -> torch.Tensor:
    """
    There is some kind of redundancy in parameters. It's intentional and try to alleviate it by some assertions
    :param interp:
    :param u_step:
    :param u_levels:
    :param Yq:
    :param N:
    :return:
    """
    # interpolation methods
    interp_methods = ["linear", "cubic"]
    # some assertions for params
    assert interp in interp_methods, f"interp param must be one of {interp_methods}"
    eps = 1e-6
    Yq_size = Yq.size()
    assert len(Yq_size) == 1, "Yq must be of 1 dim"
    u_levels_size = u_levels.size()
    assert (len(u_levels.size()) == 1), "u_levels must be of dim 1"
    assert u_levels_size[0] == Yq_size[0], "Yq must have same 1-dim size as u_levels"
    assert np.abs(u_levels[0] - 0) <= eps, "u_levels[0] must be 0"
    assert np.abs(u_levels[-1] - 1) <= eps, "u_levels[-1] must be 1"
    assert np.abs(u_step - 1.0 / (u_levels_size[0] - 1)) <= eps, "u_levels size must compatible with u_step"
    #
    u_sample = torch.distributions.Uniform(0, 1).sample(torch.Size([N]))
    Y_sample = None
    if interp == 'linear':
        idx_low = torch.floor(u_sample / u_step).type(torch.int)
        idx_high = idx_low + 1
        u_low = u_levels[idx_low]
        u_high = u_levels[idx_high]
        Yq_low = Yq[idx_low]
        Yq_high = Yq[idx_high]
        m = (u_sample - u_low) / (u_high - u_low)
        Y_sample = torch.mul(m, Yq_high - Yq_low) + Yq_low
    elif interp == 'cubic':
        cs = CubicSpline(x=u_levels.detach().numpy(), y=Yq.detach().numpy())
        Y_sample_np = cs(u_sample.detach().numpy())
        Y_sample = torch.tensor(Y_sample_np, dtype=torch.float32)
    else:
        raise ValueError(f'Invalid interp param val = {interp} : must be one of {interp_methods}')
    return Y_sample


def check_wd(wd: float, wd_thresh: float):
    if wd <= wd_thresh:
        print(f"wd is small enough : {wd} <= {wd_thresh}")
    else:
        print(f'wd is large  = {wd} >= {wd_thresh}')


if __name__ == '__main__':
    D = 4
    N = 10000
    eps = 1e-6
    wd_thresh = 1e-2
    u_step = 1e-5
    ##
    u_levels = torch.arange(start=0, end=1 + u_step, step=u_step)

    ## Test Case 1: MVN with Diagonal Covariance

    target_dist_mean = torch.distributions.Uniform(-10, 10).sample(torch.Size([D]))
    A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D]))
    target_dist_cov = torch.matmul(A, A.T)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)
    print(f'Running test case 1 with target_dist = {target_dist}\n'
          f'mean = {target_dist.mean}\n'
          f'covariance matrix = {target_dist.covariance_matrix}')

    Y = target_dist.sample(torch.Size([N]))
    sample_mean_direct = torch.mean(Y, dim=0)
    Yq = torch.quantile(Y, dim=0, q=u_levels)
    sample_inv = [uv_sample(Yq=Yq[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step, interp='cubic')
                  for i in range(D)]
    # Validate using MSE for mean and variance

    sample_inv_tensor = torch.stack(sample_inv, dim=1)
    sample_mean_inv = torch.mean(sample_inv_tensor, dim=0)
    print(f'Inv sample vs direct sample mean mse = {MSELoss()(sample_mean_inv, sample_mean_direct)}')
    sample_cov_direct = torch.cov(Y.T)
    sample_cov_inv = torch.cov(sample_inv_tensor.T)
    print(f'Inv sample vs direct sample cov mse = {MSELoss()(sample_cov_direct, sample_cov_inv)}')
    # Validate using normality test
    norm_test_direct_sample = pg.multivariate_normality(X=Y.detach().numpy())
    norm_test_inv_sample = pg.multivariate_normality(X=sample_inv_tensor.detach().numpy())
    print(f'MVN HZ test for direct sample = {norm_test_direct_sample}')
    print(f'MVN HZ test for inv. sample = {norm_test_inv_sample}')
    # use wasserstein distance
    # wd for the target distribution against a direct sample: to be used as a baseline
    wd_direct = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                   m2=sample_mean_direct, C2=sample_cov_direct)
    # the wd for direct sample should be small
    print(f'checking wd direct sample')
    check_wd(wd=wd_direct, wd_thresh=wd_thresh)
    # wd for inv sample
    print(f'checking wd inv sample')
    wd_inv = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                m2=sample_mean_inv, C2=sample_cov_inv)
    check_wd(wd=wd_inv, wd_thresh=wd_thresh)
    print('Finished Test Case 1\n***')
    # Test Case 2 - MVN -> ICA -> Y_ICA_quantiles -> inv-sample -> inv-ICA -> back to MVN
    print(f'Running Test Case 2 : Using ICA for target-dist = {target_dist}')
    transformer = FastICA(tol=1e-3, max_iter=1000)
    Y_ICA = torch.tensor(transformer.fit_transform(Y.detach().numpy()))
    print(f'Y_ICA mean = {torch.mean(Y_ICA, dim=0)}')
    print(f'Y_ICA cov = {torch.cov(Y_ICA.T)}')
    Y_ICA_q = torch.quantile(Y_ICA, dim=0, q=u_levels)
    sample_qinv_ica = torch.stack(
        [uv_sample(Yq=Y_ICA_q[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step, interp='cubic')
         for i in range(D)]).T
    print(
        f'MSE for means of Y_ICA and its Q-Inv sample ='
        f' {MSELoss()(torch.mean(Y_ICA, dim=0), torch.mean(sample_qinv_ica, dim=0))}')
    print(
        f'MSE for Cov of Y_ICA and its Q-Inv sample = '
        f'{MSELoss()(torch.cov(Y_ICA.T), torch.cov(sample_qinv_ica.T))}')
    Y_recons = torch.tensor(transformer.inverse_transform(sample_qinv_ica.detach().numpy()), dtype=torch.float32)
    wd_ica_qinv_recons = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                            m2=torch.mean(Y_recons, dim=0), C2=torch.cov(Y_recons.T))
    print(f'checking wd_ica_qinv_recons ')
    check_wd(wd=wd_ica_qinv_recons, wd_thresh=wd_thresh)
    print(f'Finished Test Case 2\n***')
