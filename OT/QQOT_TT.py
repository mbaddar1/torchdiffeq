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
# 0, 42, 1234
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
    # N = 10000
    batch_size = 32000
    validation_sample_size = batch_size
    n_data_iter = 1
    p_step = 1e-4
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
    # target_dist_cov = torch.diag(torch.distributions.Uniform(0.9, 5).sample(torch.Size([D])))
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)
    # TODO
    # test re-construct-ability of the target dist using self-QQ-reg, PCA AND q-inv

    print(f'base dist  ={base_dist}, mean = {base_dist.mean}, cov = {base_dist.covariance_matrix}')
    print(f'target dist = {target_dist}, mean = {target_dist.mean}, cov = {target_dist.covariance_matrix}')

    transformer = IncrementalPCA(whiten=True)
    # prepare training data for TT
    Xq_list = []
    Yq_list = []
    for i in range(n_data_iter):
        X_batch = base_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        Y_batch = target_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        transformer.partial_fit(Y_batch.detach().numpy())
        Y_batch_comp = torch.tensor(transformer.transform(Y_batch.detach().numpy()), dtype=torch_dtype,
                                    device=torch_device)

        Xq_batch = torch.quantile(input=X_batch, dim=0, q=p_levels)
        Xq_batch = torch.cat([Xq_batch, p_levels.view(-1, 1)], dim=1)
        Yq_batch_comp = torch.quantile(input=Y_batch_comp, dim=0, q=p_levels)
        Xq_list.append(Xq_batch)
        Yq_list.append(Yq_batch_comp)
    Xq_all = torch.cat(tensors=Xq_list, dim=0)
    Yq_all = torch.cat(tensors=Yq_list, dim=0)
    Xq_all_domain_adjusted = domain_adjust(x=Xq_all, domain_stripe=[-1, 1])
    N = Xq_all.size()[0]
    N_train = N # N - len(p_levels)
    train_index = torch.range(0, N_train - 1).type(torch.int32)
    #test_index = torch.range(N_train, N - 1).type(torch.int32)

    Xq_train = torch.index_select(input=Xq_all, index=train_index, dim=0)
    Yq_train = torch.index_select(input=Yq_all, index=train_index, dim=0)
    Xq_test = torch.index_select(input=Xq_all, index=train_index, dim=0)
    Yq_test = torch.index_select(input=Yq_all, index=train_index, dim=0)

    model = get_ETTs(D_in=D + 1, D_out=D, rank=3, domain_stripe=[-1, 1], poly_degree=6, device=torch_device)
    run_tt_als(x=Xq_train, y=Yq_train, ETT_fits=model, test_ratio=0.2, tol=1e-6, domain_stripe=[-1, 1],
               max_iter=60, regularization_coeff=float(1000))

    Yq_pred_in_sample = ETT_fits_predict(x=Xq_train, ETT_fits=model, domain_stripe=[-1, 1])
    mse_loss_in_sample = MSELoss()(Yq_train, Yq_pred_in_sample)
    print(f'MSE QQ in-sample = {mse_loss_in_sample}')

    Yq_test_out_of_sample = ETT_fits_predict(x=Xq_test, ETT_fits=model, domain_stripe=[-1, 1])
    mse_loss_out_of_sample = MSELoss()(Yq_test, Yq_test_out_of_sample)
    print(f'MSE QQ one-batch out of-sample = {mse_loss_out_of_sample}')
    Y_comp_qinv_sample = torch.stack([
        uv_sample(Yq=Yq_test_out_of_sample[:, i].reshape(-1), N=4096, u_levels=p_levels, u_step=p_step, interp='cubic')
        for i in range(D)]).T.type(torch_dtype).to(torch_device)
    Y_recons = torch.tensor(transformer.inverse_transform(Y_comp_qinv_sample.detach().numpy()))
    mean_Y_recons = torch.mean(Y_recons, dim=0)
    cov_Y_recons = torch.cov(Y_recons.T)
    wd = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_Y_recons, C1=target_dist.covariance_matrix,
                                            C2=cov_Y_recons)
    mvn_hz_test = pg.multivariate_normality(X=Y_recons.detach().numpy())
    print(f'wd = {wd}')
    print(f'mvn_hz_test = {mvn_hz_test}')
    print('finished')
