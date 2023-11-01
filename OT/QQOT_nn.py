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
    validate_qq_model

# Common Seeds
# https://www.kaggle.com/code/residentmario/kernel16e284dcb7
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# https://www.residentmar.io/2016/07/08/randomly-popular.html
# Working seed values
# 0, 42, 1234
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
    # N = 10000
    batch_size = 8192
    validation_sample_size = batch_size
    niter = 3000
    p_step = 1e-4
    torch_dtype = torch.float64
    torch_device = torch.device('cpu')
    p_levels = torch.tensor(list(np.arange(0, 1 + p_step, p_step)), dtype=torch_dtype, device=torch_device)
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.distributions.Uniform(-0.05, 0.05).sample(torch.Size([D])).type(torch_dtype).to(torch_device),
        covariance_matrix=torch.diag(
            torch.distributions.Uniform(0.1, 0.9).sample(torch.Size([D])).type(torch_dtype).to(torch_device)))
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
    print(f'Start Training')

    learning_rate = 0.2
    model = Reg(in_dim=D, out_dim=D, hidden_dim=50, bias=True, model_type='nonlinear',
                torch_device=torch_device, torch_dtype=torch_dtype)
    print(f'model = {model}')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    # main training loop
    print(f'Starting Training...')
    start_time = datetime.datetime.now()
    for i in range(niter):
        X_batch = base_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        Y_batch = target_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        transformer.partial_fit(Y_batch.detach().numpy())
        Y_batch_comp = torch.tensor(transformer.transform(Y_batch.detach().numpy()), dtype=torch_dtype,
                                    device=torch_device)

        Xq_batch = torch.quantile(input=X_batch, dim=0, q=p_levels)
        # Xq_batch_aug = torch.cat([Xq_batch, u_levels.view(-1, 1)], dim=1)
        Yq_batch_comp = torch.quantile(input=Y_batch_comp, dim=0, q=p_levels)

        optimizer.zero_grad()
        Yq_hat = model(Xq_batch)
        loss = MSELoss()(Yq_batch_comp, Yq_hat)
        losses.append(loss.item())
        print(f'i = {i + 1} , avg-running-loss = {np.mean(losses[-10:])}')
        loss.backward()
        optimizer.step()

    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).seconds
    print(f'Training time = {training_time} seconds')

    res = validate_qq_model(base_dist=base_dist, target_dist=target_dist, model=model, p_levels=p_levels,
                            N=validation_sample_size,
                            train_transformer=transformer, repeats=5, p_step=p_step, D=D, torch_dtype=torch_dtype,
                            torch_device=torch_device)
    print(f'Validation Results :\n{res}')

    print('finished')
"""
One promising case
Validation Results
1) 
{'ica_qq_mses': [0.001005039899609983, 0.0008140808204188943, 0.0004935091128572822, 0.0009460521978326142, 0.000420206954004243], 'cdf_mses': [tensor([7.1552e-05, 3.2046e-05, 2.8492e-05, 3.2107e-05], dtype=torch.float64), tensor([1.6845e-05, 4.5508e-05, 1.3698e-05, 1.6055e-05], dtype=torch.float64), tensor([2.9960e-05, 1.4597e-05, 3.2163e-05, 1.6161e-05], dtype=torch.float64), tensor([5.0877e-05, 3.0001e-05, 3.5024e-05, 4.6587e-05], dtype=torch.float64), tensor([4.3011e-05, 1.6869e-05, 1.6139e-05, 1.0125e-05], dtype=torch.float64)], 'wd': [{'baseline': 0.026816830039024353, 'reconstruct': 0.05305531620979309}, {'baseline': 0.02705707773566246, 'reconstruct': 0.029273007065057755}, {'baseline': 0.011196817271411419, 'reconstruct': 0.05634298920631409}, {'baseline': 0.021193064749240875, 'reconstruct': 0.05442590266466141}, {'baseline': 0.04101702570915222, 'reconstruct': 0.08220994472503662}], 'mvn_hz': [{'baseline': HZResults(hz=0.9993138160874909, pval=0.270056437280604, normal=True), 'reconstruct': HZResults(hz=0.9951571486630992, pval=0.29706320424050053, normal=True)}, {'baseline': HZResults(hz=1.0136730245507726, pval=0.18788980646604825, normal=True), 'reconstruct': HZResults(hz=0.9924053621513957, pval=0.3156602726084681, normal=True)}, {'baseline': HZResults(hz=1.036639900055987, pval=0.09433888620444575, normal=True), 'reconstruct': HZResults(hz=1.0159619011791428, pval=0.17647896963474963, normal=True)}, {'baseline': HZResults(hz=1.0896495908160366, pval=0.011633926675290313, normal=False), 'reconstruct': HZResults(hz=1.0471832248829231, pval=0.06575460587627185, normal=True)}, {'baseline': HZResults(hz=0.9238235297776269, pval=0.8134571474274306, normal=True), 'reconstruct': HZResults(hz=1.0060940125600648, pval=0.2290251421017136, normal=True)}]}

"""
