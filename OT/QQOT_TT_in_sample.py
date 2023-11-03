import datetime
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
import numpy as np
from sklearn.decomposition import PCA, FastICA, IncrementalPCA
import pingouin as pg
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

from OT.models import Reg
from OT.utils import wasserstein_distance_two_gaussians, get_ETTs, run_tt_als, ETT_fits_predict, uv_sample, \
    validate_qq_model, domain_adjust, get_base_dist_quantiles

# Common Seeds
# https://www.kaggle.com/code/residentmario/kernel16e284dcb7
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# https://www.residentmar.io/2016/07/08/randomly-popular.html
# Working seed values
# SEEDS to test : 0, 1, 10, 42, 123, 1234, 12345
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

"""
Flexible Generic Distributions 

https://en.wikipedia.org/wiki/Metalog_distribution 

"""

if __name__ == '__main__':
    D = 4
    train_sample_size = 32768
    test_sample_size = 4096
    p_step_train = 1e-4
    p_step_test = p_step_train  # / 2.0
    eps = 1e-8
    assert eps < p_step_test
    torch_dtype = torch.float64
    torch_device = torch.device('cpu')
    lowest_p_val = eps
    highest_p_val = 1.0
    p_levels_in_sample = torch.arange(lowest_p_val, highest_p_val + eps, p_step_train).type(torch_dtype).to(
        torch_device)
    p_levels_out_of_sample = p_levels_in_sample + p_step_train/100
    # torch.arange(start=lowest_p_val, end=highest_p_val + eps, step=p_step_test).type(torch_dtype).to(torch_device)
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.distributions.Uniform(-0.05, 0.05).sample(torch.Size([D])).type(torch_dtype).to(torch_device),
        covariance_matrix=torch.diag(
            torch.distributions.Uniform(0.1, 0.5).sample(torch.Size([D])).type(torch_dtype).to(torch_device)))
    target_dist_mean = torch.distributions.Uniform(-10, 10).sample(torch.Size([D])).type(torch_dtype).to(torch_device)
    A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D])).type(torch_dtype).to(torch_device)
    target_dist_cov = torch.matmul(A, A.T)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)
    # TODO

    print(f'base dist  ={base_dist}, mean = {base_dist.mean}, cov = {base_dist.covariance_matrix}')
    print(f'target dist = {target_dist}, mean = {target_dist.mean}, cov = {target_dist.covariance_matrix}')
    # get training data
    transformer = PCA(whiten=True)
    X_train = base_dist.sample(torch.Size([train_sample_size]))
    Xq_train = get_base_dist_quantiles(p_levels=p_levels_in_sample, base_dist=base_dist, torch_dtype=torch_dtype,
                                       torch_device=torch_device)
    # Xq = torch.quantile(input=X_sample, q=p_levels, dim=0)
    Xq_train = torch.cat([Xq_train, p_levels_in_sample.view(-1, 1)], dim=1)
    Y_train = target_dist.sample(torch.Size([train_sample_size]))
    Y_train_comp = torch.tensor(transformer.fit_transform(Y_train.detach().numpy()), dtype=torch_dtype,
                                device=torch_device)
    Yq_comp_train = torch.quantile(input=Y_train_comp, dim=0, q=p_levels_in_sample)
    model = get_ETTs(D_in=D + 1, D_out=D, rank=3, domain_stripe=[-1, 1], poly_degree=4, device=torch_device)
    start_time = datetime.datetime.now()
    run_tt_als(x=Xq_train, y=Yq_comp_train, ETT_fits=model, test_ratio=0.2, tol=1e-6, domain_stripe=[-1, 1],
               max_iter=50, regularization_coeff=float(1e4))
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).seconds
    print(f'Training time = {training_time} secs')

    ### Testing ###
    # QQ Reg test via MSE
    # In sample
    Yq_comp_pred_in_sample = ETT_fits_predict(ETT_fits=model, x=Xq_train, domain_stripe=[-1, 1])
    mse_in_sample = MSELoss()(Yq_comp_pred_in_sample, Yq_comp_train)
    print(f'QQ mse in sample = {mse_in_sample}')

    # Out of sample

    Xq_test = get_base_dist_quantiles(p_levels=p_levels_out_of_sample, base_dist=base_dist, torch_dtype=torch_dtype,
                                      torch_device=torch_device)
    Xq_test = torch.cat([Xq_test, p_levels_out_of_sample.view(-1, 1)], dim=1)
    Yq_comp_pred_out_of_sample = ETT_fits_predict(ETT_fits=model, x=Xq_test, domain_stripe=[-1, 1])
    Y_test_sample = target_dist.sample(torch.Size([test_sample_size]))
    Y_test_sample_comp = torch.tensor(PCA(whiten=True).fit_transform(Y_test_sample)).type(torch_dtype).to(torch_device)
    Yq_comp_test = torch.quantile(Y_test_sample_comp, dim=0, q=p_levels_out_of_sample)
    mse_out_of_sample = MSELoss()(Yq_comp_pred_out_of_sample, Yq_comp_test)
    print(f'QQ mse out of sample = {mse_out_of_sample}')

    ## Reconstruction Test ##

    # first set benchmark
    mean_benchmark = torch.mean(Y_test_sample, dim=0)
    cov_benchmark = torch.cov(Y_test_sample.T)
    wd_benchmark = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_benchmark,
                                                      C1=target_dist.covariance_matrix, C2=cov_benchmark)
    mvn_hz_benchmark = pg.multivariate_normality(Y_test_sample.detach().numpy())
    print(f'wd benchmark = {wd_benchmark}')
    print(f'mvn-hz benchmark = {mvn_hz_benchmark}')
    print('---')
    # i) in-sample re-construction test
    Y_comp_qinv_in_sample = torch.stack([
        uv_sample(Yq=Yq_comp_pred_in_sample[:, d].reshape(-1), N=test_sample_size, p_levels=p_levels_in_sample,
                  p_step=p_step_test,
                  interp='cubic')
        for d in range(D)]).T.type(torch_dtype).to(torch_device)
    Y_recons_in_sample = torch.tensor(transformer.inverse_transform(Y_comp_qinv_in_sample.detach().numpy())).type(
        torch_dtype).to(torch_device)
    mean_recons_in_sample = torch.mean(Y_recons_in_sample, dim=0)
    cov_recons_in_sample = torch.cov(Y_recons_in_sample.T)

    wd_in_sample = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_recons_in_sample,
                                                      C1=target_dist.covariance_matrix,
                                                      C2=cov_recons_in_sample)

    mvn_hz_in_sample = pg.multivariate_normality(Y_recons_in_sample.detach().numpy())
    print(f'wd in-sample = {wd_in_sample}')
    print(f'mvn-hz in-sample = {mvn_hz_in_sample}')
    print('---')
    # ii) out-of-sample reconstruction test
    Y_comp_qinv_out_of_sample = torch.stack([
        uv_sample(Yq=Yq_comp_pred_out_of_sample[:, d].reshape(-1), N=test_sample_size, p_levels=p_levels_out_of_sample,
                  p_step=p_step_test,
                  interp='cubic')
        for d in range(D)]).T.type(torch_dtype).to(torch_device)
    Y_recons_out_of_sample = torch.tensor(
        transformer.inverse_transform(Y_comp_qinv_out_of_sample.detach().numpy())).type(
        torch_dtype).to(torch_device)
    mean_recons_out_of_sample = torch.mean(Y_recons_out_of_sample, dim=0)
    cov_recons_out_of_sample = torch.cov(Y_recons_out_of_sample.T)

    wd_out_of_sample = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_recons_out_of_sample,
                                                          C1=target_dist.covariance_matrix,
                                                          C2=cov_recons_out_of_sample)

    mvn_hz_out_of_sample = pg.multivariate_normality(Y_recons_out_of_sample.detach().numpy())
    print(f'wd out-of-sample = {wd_out_of_sample}')
    print(f'mvn-hz out-of-sample = {mvn_hz_out_of_sample}')
    print('---')