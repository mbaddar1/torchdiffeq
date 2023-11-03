import datetime
import random

import numpy as np
import pingouin as pg
import torch
from sklearn.decomposition import IncrementalPCA
from torch.nn import MSELoss

from OT.utils import wasserstein_distance_two_gaussians, get_ETTs, run_tt_als, ETT_fits_predict, uv_sample, \
    get_train_data, get_test_data

# Common Seeds
# https://www.kaggle.com/code/residentmario/kernel16e284dcb7
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# https://www.residentmar.io/2016/07/08/randomly-popular.html

# https://prime-numbers.fandom.com/wiki/Category:4-Digit_Prime_Numbers
SEEDS = [42, 100003, 100005]
SEED = 100005
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    adjust_domain_flag = False
    max_tt_als_itr = 200
    #
    base_dist = torch.distributions.MultivariateNormal(
        loc=torch.distributions.Uniform(-0.05, 0.05).sample(torch.Size([D])).type(torch_dtype).to(torch_device),
        covariance_matrix=torch.diag(
            torch.distributions.Uniform(0.1, 0.5).sample(torch.Size([D])).type(torch_dtype).to(torch_device)))
    target_dist_mean = torch.distributions.Uniform(-10, 10).sample(torch.Size([D])).type(torch_dtype).to(torch_device)
    A = torch.distributions.Uniform(-5.0, 5.0).sample(torch.Size([D, D])).type(torch_dtype).to(torch_device)
    target_dist_cov = torch.matmul(A, A.T)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)

    # Get training and test Data
    data_size = 4096
    num_batches_train = 8
    num_p_levels = int(1e3)
    train_transformer = IncrementalPCA(whiten=True)
    test_transformer = IncrementalPCA(whiten=True)
    Xq_train, Yq_train = get_train_data(base_dist=base_dist, target_dist=target_dist, data_batch_size=data_size,
                                        num_batches=num_batches_train, num_p_levels=num_p_levels,
                                        torch_dtype=torch_dtype, torch_device=torch_device,
                                        transformer=train_transformer)
    Xq_test, Yq_test = get_test_data(base_dist=base_dist, target_dist=target_dist,
                                     num_p_levels=num_p_levels * 10,
                                     torch_dtype=torch_dtype, torch_device=torch_device, data_size=data_size)
    # Modeling
    model = get_ETTs(D_in=D + 1, D_out=D, rank=4, domain_stripe=[-5, 5], poly_degree=6, device=torch_device)
    start_time = datetime.datetime.now()
    run_tt_als(x=Xq_train, y=Yq_train, ETT_fits=model, test_ratio=0.2, tol=1e-6, domain_stripe=[-1, 1],
               max_iter=max_tt_als_itr, regularization_coeff=float(1e-2), adjust_domain_flag=adjust_domain_flag)
    end_time = datetime.datetime.now()
    Yq_pred = ETT_fits_predict(ETT_fits=model, x=Xq_test, domain_stripe=[-1, 1], adjust_domain_flag=adjust_domain_flag)
    mse = MSELoss()(Yq_pred, Yq_test)
    print(f'mse for out of sample prediction = {mse}')

    # Reconstruction Error
    p_levels_test = Xq_test[:, D]  # p_level is augmented as last column to X
    Y_comp_qinv = torch.stack([
        uv_sample(Yq=Yq_pred[:, d].reshape(-1), N=test_sample_size, p_levels=p_levels_test,
                  interp='cubic') for d in range(D)]).T.type(torch_dtype).to(torch_device)
    Y_recons = torch.tensor(train_transformer.inverse_transform(Y_comp_qinv.detach().numpy()))
    mean_recons = torch.mean(Y_recons, dim=0)
    cov_recons = torch.cov(Y_recons.T)
    wd = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_recons, C1=target_dist.covariance_matrix,
                                            C2=cov_recons)
    print(f'training time = {(end_time - start_time).seconds}')
    print(f'wd = {wd}')
    print(f'mvn-hz = {pg.multivariate_normality(X=Y_recons.detach().numpy())}')
