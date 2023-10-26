import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
import numpy as np
from sklearn.decomposition import PCA, FastICA
import pingouin as pg
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

from OT.inverse_transform_samples_quantile_interp import uv_sample
from OT.utils import wasserstein_distance_two_gaussians

"""
Flexible Generic Distributions 
https://en.wikipedia.org/wiki/Metalog_distribution 

"""


class Reg(torch.nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int, type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type == 'linear':
            self.model = torch.nn.Linear(in_out_dim, in_out_dim)
        elif type == 'nonlinear':
            self.model = torch.nn.Sequential(torch.nn.Linear(in_out_dim, hidden_dim),
                                             torch.nn.ReLU(),
                                             # torch.nn.Linear(hidden_dim, hidden_dim),
                                             # torch.nn.ReLU(),
                                             torch.nn.Linear(hidden_dim, in_out_dim)
                                             )

    def forward(self, x):
        return self.model(x)


def univariate_inv_sample(Yq_d: torch.Tensor, u: torch.Tensor):
    N = Yq_d.size()[0]
    step = 1.0 / (N - 1)
    i = torch.floor(u / step).type(torch.int)
    u1 = i * step
    u2 = u1 + step
    m = (u - u1) / (u2 - u1)
    s1 = Yq_d[i]
    s2 = Yq_d[i + 1]
    s = m * (s2 - s1) + s1
    return s


def multivariate_inv_sample(Yq: torch.Tensor, N_samples: int):
    D = Yq.size()[1]
    N = Yq.size()[0]
    eps = 1e-4
    u = torch.distributions.Uniform(0 + eps, 1 - eps).sample(torch.Size([N_samples])).view(-1, 1)
    samples_list = []
    for d in range(D):
        Yq_d = Yq[:, d]
        samples = torch.vmap(lambda x: univariate_inv_sample(Yq_d=Yq_d, u=x), in_dims=0)(u)
        samples_list.append(samples)

    # internal assertion
    for j in range(len(samples_list)):
        Yq_ref = Yq[:, j]
        Yq_d_est = torch.quantile(q=torch.tensor(list(np.arange(0, 1 + eps, 1.0 / (N - 1))), dtype=torch.float32),
                                  input=samples_list[j], dim=0)
        abs_diff = torch.abs(Yq_ref - Yq_d_est)
        mse_ = MSELoss()(Yq_d_est, Yq_ref)
    return torch.cat(samples_list, dim=1)


def validate_qq_model(base_dist: torch.distributions.Distribution,
                      target_distribution: torch.distributions.Distribution, model: torch.nn.Module, N: int,
                      q: torch.Tensor, transformer: FastICA, repeats: int) -> dict:
    mses_qq = []
    mse_cdfs = []
    wd_list = []
    mvn_hz_test_res_list = []
    for i in range(repeats):
        print(f'validation iteration {i + 1} out of {repeats}')
        # q-q validation
        X_test = base_dist.sample(torch.Size([N]))
        Xq_test = torch.quantile(input=X_test, q=q, dim=0)
        Y_test = target_distribution.sample(torch.Size([N]))
        Y_test_ICA = torch.tensor(transformer.fit_transform(Y_test.detach().numpy()))
        Yq_ICA_test_ref = torch.quantile(input=Y_test_ICA, dim=0, q=q)
        Yq_pred = model(Xq_test)

        mse = MSELoss()(Yq_ICA_test_ref, Yq_pred).item()
        mses_qq.append(mse)
        # cdf validation
        for j in range(D):
            plt.plot(q.detach().numpy(), Yq_ICA_test_ref[:, j].detach().numpy())
            plt.plot(q.detach().numpy(), Yq_pred[:, j].detach().numpy())
            plt.savefig(f'cdf_d_{j}.png')
            plt.clf()
            plt.plot(Yq_ICA_test_ref[:, j].detach().numpy(), Yq_ICA_test_ref[:, j].detach().numpy())
            plt.plot(Yq_ICA_test_ref[:, j].detach().numpy(), Yq_pred[:, j].detach().numpy())
            plt.savefig(f'qq_d_{j}.png')
            plt.clf()
        mse_cdf_repeat = []
        for j in range(D):
            y_ica_j = Yq_ICA_test_ref[:, j].detach().numpy()
            ecdf = ECDF(x=y_ica_j)
            cdf_ref = ecdf(Yq_ICA_test_ref[:, j].detach().numpy())
            cdf_est = ecdf(Yq_pred[:, j].detach().numpy())
            plt.clf()
            mse_cdf_j = MSELoss()(torch.tensor(cdf_ref), torch.tensor(cdf_est))
            mse_cdf_repeat.append(mse_cdf_j)
        mse_cdfs.append(torch.tensor(mse_cdf_repeat))
        # validate the reconstruction process
        if isinstance(target_distribution, torch.distributions.MultivariateNormal):
            Y_ica_qinv_sample = torch.stack([
                uv_sample(Yq=Yq_ICA_test_ref[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step, interp='cubic')
                for i in range(D)]).T.type(torch.float32)
            # FIXME remove : set of debug vars
            # m1 = torch.mean(Y_test_ICA ,dim=0)
            # m2 = torch.mean(Y_ica_qinv_sample,dim=0)
            # mse_m_ = MSELoss()(m1,m2)
            # cov1 = torch.cov(Y_test_ICA.T)
            # cov2 = torch.cov(Y_ica_qinv_sample.T)
            # mse_cov_ = MSELoss()(cov1,cov2)
            Y_recons = torch.tensor(transformer.inverse_transform(Y_ica_qinv_sample.detach().numpy()))
            wd_baseline = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                             m2=torch.mean(Y_test, dim=0), C2=torch.cov(Y_test.T))
            wd_recons = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                           m2=torch.mean(Y_recons, dim=0), C2=torch.cov(Y_recons.T))
            mvn_hz_baseline = pg.multivariate_normality(X=Y_test.detach().numpy())
            mvn_hz_recons = pg.multivariate_normality(X=Y_recons.detach().numpy())
            wd_list.append({'baseline': wd_baseline, 'reconstruct': wd_recons})
            mvn_hz_test_res_list.append({'baseline': mvn_hz_baseline, 'reconstruct': mvn_hz_recons})

    res = dict()
    res['ica_qq_mses'] = mses_qq
    res['cdf_mses'] = mse_cdfs
    res['wd'] = wd_list
    res['mvn_hz'] = mvn_hz_test_res_list
    return res


if __name__ == '__main__':
    D = 4
    N = 10000
    batch_size = 1024
    niter = 150
    u_step = 1e-5
    u_levels = torch.tensor(list(np.arange(0, 1 + u_step, u_step)), dtype=torch.float32)
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.distributions.Uniform(-0.05, 0.05).sample(torch.Size([D])),
        covariance_matrix=torch.diag(
            torch.distributions.Uniform(0.1, 0.9).sample(torch.Size([D]))))
    target_dist_mean = torch.distributions.Uniform(-10, 10).sample(torch.Size([D]))
    A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D]))
    target_dist_cov = torch.matmul(A, A.T)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)
    print(f'base dist  ={base_dist}, mean = {base_dist.mean}, cov = {base_dist.covariance_matrix}')
    print(f'target dist = {target_dist}, mean = {target_dist.mean}, cov = {target_dist.covariance_matrix}')
    # Original Samples
    X = base_dist.sample(torch.Size([N]))
    Y = target_dist.sample(torch.Size([N]))
    # Apply ICA
    transformer = FastICA(random_state=0, whiten='unit-variance', max_iter=2000, tol=1e-4)
    Y_ICA = torch.tensor(transformer.fit_transform(X=Y), dtype=torch.float32)
    # apply Qinv to Y_ICA and see if it works
    Y_ICA_q = torch.quantile(input=Y_ICA, q=u_levels, dim=0)
    Y_ICA_qinv = torch.stack(
        [uv_sample(Yq=Y_ICA_q[:, i].reshape(-1), N=N, u_levels=u_levels, u_step=u_step, interp='cubic')
         for i in range(D)]).T
    # m1 = torch.mean(Y_ICA, dim=0)
    # m2 = torch.mean(Y_ICA_qinv, dim=0)
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
    Y_recons = torch.tensor(transformer.inverse_transform(Y_ICA.detach().numpy()))
    print(
        f'MSE for reconstructed vs actual sample mean = {MSELoss()(torch.mean(Y, dim=0), torch.mean(Y_recons, dim=0))}')
    print(f'MSE for reconstructed vs actual sample cov  = {MSELoss()(torch.cov(Y.T), torch.cov(Y_recons.T))}')
    target_sample_mvn_test = pg.multivariate_normality(X=Y)
    assert target_sample_mvn_test.normal, f"Raw target sample failed the mvn test , res = {target_sample_mvn_test}"
    target_sample_ica_mvn_test = pg.multivariate_normality(X=Y_ICA)
    print(f'Normality test for the actual sample = {target_sample_mvn_test}')
    print(f'Normality test for the reconstructed sample = {target_sample_ica_mvn_test}')
    # create empirical cdf functions
    ecdfs = []

    for i in range(D):
        y_ica_i = Y_ICA[:, i].detach().numpy()
        ecdf = ECDF(x=y_ica_i)
        ecdfs.append(ecdf)

    #
    # set targets and predictors
    Yq = torch.quantile(input=Y_ICA, q=u_levels, dim=0)
    Xq = torch.quantile(input=X, q=u_levels, dim=0)
    # test the ecdf
    cdfs_ref = []
    for i in range(D):
        cdfs_ref.append(ecdfs[i](Yq[:, i]))
    #
    print(f'Start Training')
    learning_rate = 0.2
    model = Reg(in_out_dim=D, hidden_dim=100, type='nonlinear')
    print(f'model = {model}')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(niter):
        # indices = torch.randperm(n=batch_size)
        X_batch = base_dist.sample(torch.Size([batch_size]))
        Y_batch = target_dist.sample(torch.Size([batch_size]))
        Y_batch_ica = torch.tensor(transformer.fit_transform(Y_batch.detach().numpy()))
        Xq_batch = torch.quantile(input=X_batch, dim=0, q=u_levels)
        Yq_batch_ica = torch.quantile(input=Y_batch_ica, dim=0, q=u_levels)
        # Xq_batch = Xq[indices, :]
        # Yq_batch = Yq[indices, :]
        optimizer.zero_grad()
        Yq_hat = model(Xq_batch)
        loss = MSELoss()(Yq_batch_ica, Yq_hat)
        print(f'i = {i} , loss = {loss}')
        loss.backward()
        optimizer.step()
    res = validate_qq_model(base_dist=base_dist, target_distribution=target_dist, model=model, q=u_levels, N=10000,
                            transformer=transformer, repeats=5)
    print(f'Validation Results :\n{res}')
    # validation using out of sample data
    # print(f'In-sample validation')
    # X_test = base_dist.sample(torch.Size([N]))
    # Xq_test = torch.quantile(input=X, q=q, dim=0)
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
