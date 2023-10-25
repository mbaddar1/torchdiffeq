"""
https://stackoverflow.com/a/44163082

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

plan
1. Implement a function that takes quantiles and generate samples
2. Test Generated samples using WD distance
3. Test on Normal, Exp, LogNormal and Empirical Dist from ICA transformation
"""
import numpy as np
import torch
from torch.nn import MSELoss


def uv_sample(Yq: torch.Tensor, N: int, u_levels: torch.Tensor, u_step: float):
    """
    There is some kind of redundancy in parameters. It's intentional and try to alleviate it by some assertions
    :param u_step:
    :param u_levels:
    :param Yq:
    :param N:
    :return:
    """
    # some assertions for params
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
    idx_low = torch.floor(u_sample / u_step).type(torch.int)
    idx_high = idx_low + 1
    u_low = u_levels[idx_low]
    u_high = u_levels[idx_high]
    Yq_low = Yq[idx_low]
    Yq_high = Yq[idx_high]
    m = (u_sample - u_low) / (u_high - u_low)
    Y_sample = torch.mul(m, Yq_high - Yq_low) + Yq_low
    return Y_sample


if __name__ == '__main__':
    D = 4
    N = 10000
    eps = 1e-6
    u_step = 1e-6
    ##
    u_levels = torch.arange(start=0, end=1 + u_step, step=u_step)
    target_dist_mean = torch.distributions.Uniform(-2, 2).sample(torch.Size([D]))
    # A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D]))
    # target_dist_cov = torch.matmul(A, A.T)
    target_dist_cov = torch.diag(torch.distributions.Uniform(0.1, 5).sample(torch.Size([D])))
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean,
                                                         covariance_matrix=target_dist_cov)
    Y = target_dist.sample(torch.Size([N]))
    sample_mean_direct = torch.mean(Y, dim=0)
    Yq = torch.quantile(Y, dim=0, q=u_levels)
    sample_inv = [uv_sample(Yq=Yq[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step) for i in range(D)]
    sample_inv_tensor = torch.stack(sample_inv,dim=1)
    sample_mean_inv = torch.mean(sample_inv_tensor,dim=0)
    print(f'Inv sample vs direct sample mean mse = {MSELoss()(sample_mean_inv,sample_mean_direct)}')
    sample_cov_direct = torch.cov(Y.T)
    sample_cov_inv = torch.cov(sample_inv_tensor.T)
    print(f'Inv sample vs direct sample cov mse = {MSELoss()(sample_cov_direct, sample_cov_inv)}')
    print('finished')
