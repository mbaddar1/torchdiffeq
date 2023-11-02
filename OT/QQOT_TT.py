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
    validate_qq_model, domain_adjust

# Common Seeds
# https://www.kaggle.com/code/residentmario/kernel16e284dcb7
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# https://www.residentmar.io/2016/07/08/randomly-popular.html
# Working seed values
# SEEDS to test : 0, 1, 10, 42, 123, 1234, 12345
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

"""
Flexible Generic Distributions 

https://en.wikipedia.org/wiki/Metalog_distribution 

"""

if __name__ == '__main__':
    D = 4
    X_sample_size = Y_sample_size = 32768
    p_step = 5e-5
    torch_dtype = torch.float64
    torch_device = torch.device('cpu')
    p_levels = torch.tensor(list(np.arange(0, 1 + p_step, p_step)), dtype=torch_dtype, device=torch_device)
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

    transformer = PCA(whiten=True)
    Xq_list = []
    Yq_list = []
    X_sample = base_dist.sample(torch.Size([X_sample_size]))
    Xq = torch.quantile(input=X_sample, q=p_levels, dim=0)
    Y_sample = target_dist.sample(torch.Size([Y_sample_size]))
    Y_comp = torch.tensor(transformer.fit_transform(Y_sample.detach().numpy()), dtype=torch_dtype, device=torch_device)
    Yq_comp = torch.quantile(input=Y_comp, dim=0, q=p_levels)
    model = get_ETTs(D_in=D, D_out=D, rank=3, domain_stripe=[-1, 1], poly_degree=6, device=torch_device)
    start_time = datetime.datetime.now()
    run_tt_als(x=Xq, y=Yq_comp, ETT_fits=model, test_ratio=0.2, tol=1e-6, domain_stripe=[-1, 1],
               max_iter=50, regularization_coeff=float(1e-1))
    end_time = datetime.datetime.now()
    training_time = (end_time-start_time).seconds
    print(f'Training time = {training_time} secs')
    # in sample test

    Yq_comp_pred = ETT_fits_predict(ETT_fits=model, x=Xq, domain_stripe=[-1, 1])
    mse_in_sample = MSELoss()(Yq_comp_pred, Yq_comp)
    print(f'QQ mse in sample = {mse_in_sample}')

    # out of sample test (1)
    Y_sample_1 = target_dist.sample(torch.Size([Y_sample_size]))
    Y_sample_1_comp = torch.tensor(PCA(whiten=True).fit_transform(Y_sample_1)).type(torch_dtype).to(torch_device)
    Yq_sample_1_comp = torch.quantile(Y_sample_1_comp, dim=0, q=p_levels)
    mse_out_of_sample_1 = MSELoss()(Yq_comp_pred, Yq_sample_1_comp)
    print(f'QQ mse out of sample 1 = {mse_out_of_sample_1}')

    # out of sample (2)
    X_sample_2 = base_dist.sample(torch.Size([X_sample_size]))
    Xq_2 = torch.quantile(X_sample_2, dim=0, q=p_levels)
    Y_sample_2 = target_dist.sample(torch.Size([Y_sample_size]))
    Y_sample_2_comp = torch.tensor(PCA(whiten=True).fit_transform(Y_sample_2)).type(torch_dtype).to(torch_device)
    Yq_sample_2_comp = torch.quantile(Y_sample_2_comp, dim=0, q=p_levels)
    Yq_pred_2 = ETT_fits_predict(ETT_fits=model, x=Xq_2, domain_stripe=[-1, 1])
    mse_out_of_sample_2 = MSELoss()(Yq_pred_2, Yq_sample_2_comp)
    print(f'QQ mse out of sample 2 = {mse_out_of_sample_2}')

    # reconstruction test
    test_sample_size = 4096
    Y_sample_3 = target_dist.sample(torch.Size([test_sample_size]))
    mean_benchmark = torch.mean(Y_sample_3,dim=0)
    cov_benchmark = torch.cov(Y_sample_3.T)
    Y_comp_qinv_sample = torch.stack([
        uv_sample(Yq=Yq_comp_pred[:, d].reshape(-1), N=test_sample_size, u_levels=p_levels, u_step=p_step, interp='cubic')
        for d in range(D)]).T.type(torch_dtype).to(torch_device)
    Y_recons = torch.tensor(transformer.inverse_transform(Y_comp_qinv_sample.detach().numpy())).type(
        torch_dtype).to(torch_device)
    mean_recons = torch.mean(Y_recons, dim=0)
    cov_recons = torch.cov(Y_recons.T)
    wd_benchmark = wasserstein_distance_two_gaussians(m1=target_dist.mean,m2=mean_benchmark,
                                                      C1=target_dist.covariance_matrix,C2=cov_benchmark)
    wd_test = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_recons, C1=target_dist.covariance_matrix,
                                            C2=cov_recons)
    mvn_hz_benchmark = pg.multivariate_normality(Y_sample_3.detach().numpy())
    mvn_hz_test = pg.multivariate_normality(Y_recons.detach().numpy())
    print(f'wd test = {wd_test}')
    print(f'wd benchmark = {wd_benchmark}')
    print(f'mvn-hz test = {mvn_hz_test}')
    print(f'mvn-hz benchmark = {mvn_hz_benchmark}')
