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

SEED = 41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# SEEDS :
"""
Flexible Generic Distributions 
https://en.wikipedia.org/wiki/Metalog_distribution 

"""

if __name__ == '__main__':
    D = 4
    # N = 10000
    batch_size = 8192
    niter = 3000  # ?? seems to be good enough for good wd_reconstruct vs baseline and to pass mvn hz test
    p_step = 1e-3
    model_classes = ['nn', 'tt']
    model_class = 'nn'
    torch_dtype = torch.float64
    torch_device = torch.device('cpu')
    assert model_class in model_classes
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
    print(f'base dist  ={base_dist}, mean = {base_dist.mean}, cov = {base_dist.covariance_matrix}')
    print(f'target dist = {target_dist}, mean = {target_dist.mean}, cov = {target_dist.covariance_matrix}')
    # Original Samples
    # X = base_dist.sample(torch.Size([N])).type(torch.float32)
    # Y = target_dist.sample(torch.Size([N])).type(torch.float32)
    # Apply ICA
    # transformer = FastICA(random_state=0, whiten='unit-variance', max_iter=2000, tol=1e-4)

    # Y_comp = torch.tensor(transformer.fit_transform(X=Y), dtype=torch.float32)
    # print(f'mean for Y_comp = {torch.mean(Y_comp, dim=0)}')
    # print(f'cov of Y_comp (Should be diagonal) = {torch.cov(Y_comp.T)}')
    # apply Qinv to Y_ICA and see if it works
    # Y_comp_q = torch.quantile(input=Y_comp, q=u_levels, dim=0)
    # Y_comp_qinv = torch.stack(
    #     [uv_sample(Yq=Y_comp_q[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step, interp='cubic')
    #      for i in range(D)]).T
    # # m1 = torch.mean(Y_ICA, dim=0)
    # # m2 = torch.mean(Y_ICA_qinv, dim=0)
    # diff_m = torch.norm(m1 - m2)
    # cov1 = torch.cov(Y_ICA.T)
    # cov2 = torch.cov(Y_ICA_qinv.T)
    # diff_cov = torch.norm(cov1 - cov2)
    """
    Quick Sanity check 
    1. Reconstruct the original Y
    2. Test mean , cov and normality 
    the step test the "Re-construct-ability" of the target distribution after ICA  
    """
    # Y_recons = torch.tensor(transformer.inverse_transform(Y_comp.detach().numpy()))
    # print(
    #     f'MSE for reconstructed vs actual sample mean = {MSELoss()(torch.mean(Y, dim=0), torch.mean(Y_recons, dim=0))}')
    # print(f'MSE for reconstructed vs actual sample cov  = {MSELoss()(torch.cov(Y.T), torch.cov(Y_recons.T))}')
    # target_direct_sample_mvn_test = pg.multivariate_normality(X=Y)
    # # FIXME , find a better way for assertion
    # # assert target_sample_mvn_test.normal, f"Raw target sample failed the mvn test , res = {target_sample_mvn_test}"
    # target_recons_test = pg.multivariate_normality(X=Y_recons)
    # print(f'Normality test for the actual sample = {target_direct_sample_mvn_test}')
    # print(f'Normality test for the reconstructed sample = {target_recons_test}')
    # # create empirical cdf functions
    # ecdfs = []
    #
    # for i in range(D):
    #     y_comp_i = Y_comp[:, i].detach().numpy()
    #     ecdf = ECDF(x=y_comp_i)
    #     ecdfs.append(ecdf)
    #
    # #
    # # set targets and predictors
    # Yq = torch.quantile(input=Y_comp, q=u_levels, dim=0)
    # Xq = torch.quantile(input=X, q=u_levels, dim=0)
    # # test the ecdf
    # cdfs_ref = []
    # for i in range(D):
    #     cdfs_ref.append(ecdfs[i](Yq[:, i]))
    # #
    transformer = IncrementalPCA(whiten=True)
    print(f'Start Training')
    if model_class == 'nn':
        learning_rate = 0.1
        model = Reg(in_dim=D, out_dim=D, hidden_dim=50, bias=True, model_type='nonlinear',
                    torch_device=torch.device('cpu'), torch_dtype=torch_dtype)
        print(f'model = {model}')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif model_class == 'tt':
        model = get_ETTs(D_in=D, D_out=D, rank=3, domain_stripe=[-1, 1], poly_degree=3, device=torch.device("cpt"))
    losses = []
    # main training loop
    for i in range(niter):
        # indices = torch.randperm(n=batch_size)
        X_batch = base_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        Y_batch = target_dist.sample(torch.Size([batch_size])).type(torch_dtype).to(torch_device)
        transformer.partial_fit(Y_batch.detach().numpy())
        Y_batch_comp = torch.tensor(transformer.transform(Y_batch.detach().numpy()), dtype=torch_dtype,
                                    device=torch_device)
        # FIXME debug vars, to remove
        m = torch.mean(Y_batch_comp, dim=0)
        C = torch.cov(Y_batch_comp.T)
        Xq_batch = torch.quantile(input=X_batch, dim=0, q=p_levels)
        # Xq_batch_aug = torch.cat([Xq_batch, u_levels.view(-1, 1)], dim=1)
        Yq_batch_comp = torch.quantile(input=Y_batch_comp, dim=0, q=p_levels)
        loss = None
        if model_class == 'nn':
            # Xq_batch = Xq[indices, :]
            # Yq_batch = Yq[indices, :]
            optimizer.zero_grad()
            Yq_hat = model(Xq_batch)

            loss = MSELoss()(Yq_batch_comp, Yq_hat)
            losses.append(loss.item())
            print(f'i = {i + 1} ,model_type = {model_class}, avg-running-loss = {np.mean(losses[-10:])}')
            loss.backward()
            optimizer.step()
        elif model_class == 'nn':
            y_hat = ETT_fits_predict(x=Xq_batch, ETT_fits=model, domain_stripe=[-1, 1])
            loss = MSELoss()(Yq_batch_comp, y_hat)
            print(f'i = {i} ,model_type = {model_class}, loss = {loss}')
            run_tt_als(x=Xq_batch, y=Y_batch, ETT_fits=model, test_ratio=0.2, tol=1e-4, domain_stripe=[-1, 1])

    res = validate_qq_model(base_dist=base_dist, target_dist=target_dist, model=model, p_levels=p_levels,
                            N=10000,
                            train_transformer=transformer, repeats=5, p_step=p_step, D=D, torch_dtype=torch_dtype,
                            torch_device=torch_device)
    print(f'Validation Results :\n{res}')
    # validation using out of sample data
    # print(f'In-sample validation')
    # X_test = base_dist.sample(torch.Size([N]))
    # Xq_test = torch.quantile(input=X, q=q, dim=0)
    # Y_test =
    # Y_test =
    # Yq_test_hat = model(Xq_test)
    # cdfs_test = []
    # for i in range(D):
    #     cdfs_test.append(ecdfs[i](Yq_test_hat[:, i].detach().numpy()))
    # cdf_ref_tensor = torch.tensor(cdfs_ref)
    # cdf_test_tensor = torch.tensor(cdfs_test)
    # samples = multivariate_inv_sample(Yq=Yq, N_samples=10000)
    # mn = torch.mean(samples,dim=0)
    # mn1 = torch.mean(Y_ICA,dim=0)
    # c = torch.cov(samples.T)
    # c1 = torch.cov(Y_ICA.T)
    # print(f'MSE cdf ref vs cdf test = {MSELoss()(cdf_ref_tensor, cdf_test_tensor)}')
    print('finished')
"""
One promising case
Validation Results
1) 
{'ica_qq_mses': [0.001005039899609983, 0.0008140808204188943, 0.0004935091128572822, 0.0009460521978326142, 0.000420206954004243], 'cdf_mses': [tensor([7.1552e-05, 3.2046e-05, 2.8492e-05, 3.2107e-05], dtype=torch.float64), tensor([1.6845e-05, 4.5508e-05, 1.3698e-05, 1.6055e-05], dtype=torch.float64), tensor([2.9960e-05, 1.4597e-05, 3.2163e-05, 1.6161e-05], dtype=torch.float64), tensor([5.0877e-05, 3.0001e-05, 3.5024e-05, 4.6587e-05], dtype=torch.float64), tensor([4.3011e-05, 1.6869e-05, 1.6139e-05, 1.0125e-05], dtype=torch.float64)], 'wd': [{'baseline': 0.026816830039024353, 'reconstruct': 0.05305531620979309}, {'baseline': 0.02705707773566246, 'reconstruct': 0.029273007065057755}, {'baseline': 0.011196817271411419, 'reconstruct': 0.05634298920631409}, {'baseline': 0.021193064749240875, 'reconstruct': 0.05442590266466141}, {'baseline': 0.04101702570915222, 'reconstruct': 0.08220994472503662}], 'mvn_hz': [{'baseline': HZResults(hz=0.9993138160874909, pval=0.270056437280604, normal=True), 'reconstruct': HZResults(hz=0.9951571486630992, pval=0.29706320424050053, normal=True)}, {'baseline': HZResults(hz=1.0136730245507726, pval=0.18788980646604825, normal=True), 'reconstruct': HZResults(hz=0.9924053621513957, pval=0.3156602726084681, normal=True)}, {'baseline': HZResults(hz=1.036639900055987, pval=0.09433888620444575, normal=True), 'reconstruct': HZResults(hz=1.0159619011791428, pval=0.17647896963474963, normal=True)}, {'baseline': HZResults(hz=1.0896495908160366, pval=0.011633926675290313, normal=False), 'reconstruct': HZResults(hz=1.0471832248829231, pval=0.06575460587627185, normal=True)}, {'baseline': HZResults(hz=0.9238235297776269, pval=0.8134571474274306, normal=True), 'reconstruct': HZResults(hz=1.0060940125600648, pval=0.2290251421017136, normal=True)}]}

"""
