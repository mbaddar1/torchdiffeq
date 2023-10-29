"""
https://stackoverflow.com/a/44163082

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html

plan
1. Implement a function that takes quantiles and generate samples Done
2. Test Generated samples using WD distance Done
3. Test on Normal, Exp, LogNormal and Empirical Dist from ICA transformation - Not Needed Now
4- Use Cubic Spline
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html Done
"""
import numpy as np
import scipy.stats
import torch
from scipy.interpolate import CubicSpline
from sklearn.decomposition import FastICA, PCA
from torch.nn import MSELoss
from scipy.stats import skew
import pingouin as pg

from OT.utils import wasserstein_distance_two_gaussians




if __name__ == '__main__':
    D = 4
    N = 100000
    eps = 1e-6
    wd_thresh = 1e-2
    p_step = 1e-6
    ##
    p_levels = torch.arange(start=0, end=1 + p_step, step=p_step)
    p_min,p_max = torch.min(p_levels),torch.max(p_levels)
    target_dist_mean = torch.distributions.Uniform(-10, 10).sample(torch.Size([D]))
    A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D]))
    target_dist_cov = torch.matmul(A, A.T)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)

    ## Test Case 1: MVN with Diagonal Covariance

    # print(f'Running test case 1 with target_dist = {target_dist}\n'
    #       f'mean = {target_dist.mean}\n'
    #       f'covariance matrix = {target_dist.covariance_matrix}')
    #
    #
    # sample_mean_direct = torch.mean(Y, dim=0)
    # Yq = torch.quantile(Y, dim=0, q=p_levels)
    # sample_inv = [uv_sample(Yq=Yq[:, i].reshape(-1), N=N, u_levels=p_levels, u_step=p_step, interp='cubic')
    #               for i in range(D)]
    # # Validate using MSE for mean and variance
    #
    # sample_inv_tensor = torch.stack(sample_inv, dim=1)
    # sample_mean_inv = torch.mean(sample_inv_tensor, dim=0)
    # print(f'Inv sample vs direct sample mean mse = {MSELoss()(sample_mean_inv, sample_mean_direct)}')
    # sample_cov_direct = torch.cov(Y.T)
    # sample_cov_inv = torch.cov(sample_inv_tensor.T)
    # print(f'Inv sample vs direct sample cov mse = {MSELoss()(sample_cov_direct, sample_cov_inv)}')
    # # Validate using normality test
    # norm_test_direct_sample = pg.multivariate_normality(X=Y.detach().numpy())
    # norm_test_inv_sample = pg.multivariate_normality(X=sample_inv_tensor.detach().numpy())
    # print(f'MVN HZ test for direct sample = {norm_test_direct_sample}')
    # print(f'MVN HZ test for inv. sample = {norm_test_inv_sample}')
    # # use wasserstein distance
    # # wd for the target distribution against a direct sample: to be used as a baseline
    # wd_direct = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
    #                                                m2=sample_mean_direct, C2=sample_cov_direct)
    # # the wd for direct sample should be small
    # print(f'checking wd direct sample')
    # check_wd(wd=wd_direct, wd_thresh=wd_thresh)
    # # wd for inv sample
    # print(f'checking wd inv sample')
    # wd_inv = wasserstein_distance_two_gaussians(m1=target_dist.mean,
    #                                             C1=target_dist.covariance_matrix.type(torch.float32),
    #                                             m2=sample_mean_inv, C2=sample_cov_inv.type(torch.float32))
    # check_wd(wd=wd_inv, wd_thresh=wd_thresh)
    # print('Finished Test Case 1\n***')
    # Test Case 2 - MVN -> ICA -> Y_ICA_quantiles -> inv-sample -> inv-ICA -> back to MVN
    Y = target_dist.sample(torch.Size([N]))
    print(f'Running Test Case 2 : Using ICA for target-dist = {target_dist}')
    # transformer = FastICA(tol=1e-3, max_iter=1000)
    transformer = PCA()
    Y_ICA = torch.tensor(transformer.fit_transform(Y.detach().numpy()))
    print(f'Y_ICA mean = {torch.mean(Y_ICA, dim=0)}')
    print(f'Y_ICA cov = {torch.cov(Y_ICA.T)}')
    Y_ICA_q = torch.quantile(Y_ICA, dim=0, q=p_levels)
    sample_qinvs_ica = torch.stack(
        [uv_sample(Yq=Y_ICA_q[:, i].reshape(-1), N=N, u_levels=p_levels, u_step=p_step, interp='cubic')
         for i in range(D)]).T
    print(
        f'MSE for means of Y_ICA and its Q-Inv sample ='
        f' {MSELoss()(torch.mean(Y_ICA, dim=0), torch.mean(sample_qinvs_ica, dim=0))}')
    print(
        f'MSE for Cov of Y_ICA and its Q-Inv sample = '
        f'{MSELoss()(torch.cov(Y_ICA.T), torch.cov(sample_qinvs_ica.T))}')
    for i in range(D):
        skew_ref = skew(a=Y_ICA[:,i].detach().numpy())
        skew_qinvs = skew(a=sample_qinvs_ica[:,i].detach().numpy())
        print(f'skewness of ICA comp : {skew_ref,skew_qinvs}')
    Y_recons = torch.tensor(transformer.inverse_transform(sample_qinvs_ica.detach().numpy()), dtype=torch.float32)
    wd_direct = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                   m2=torch.mean(Y, dim=0), C2=torch.cov(Y.T))
    print('checking wd direct')
    check_wd(wd=wd_direct,wd_thresh=wd_thresh)
    wd_ica_qinv_recons = wasserstein_distance_two_gaussians(m1=target_dist.mean,
                                                            C1=target_dist.covariance_matrix.type(torch.float32),
                                                            m2=torch.mean(Y_recons, dim=0),
                                                            C2=torch.cov(Y_recons.T).type(torch.float32))
    print(f'checking wd_ica_qinv_recons ')
    check_wd(wd=wd_ica_qinv_recons, wd_thresh=wd_thresh)
    print(f'Finished Test Case 2\n***')
