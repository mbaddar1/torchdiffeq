"""
Matching a Distribution by Matching Quantiles Estimation
https://www.tandfonline.com/doi/epdf/10.1080/01621459.2014.929522?needAccess=true
Bayesian Quantile Matching Estimation
https://arxiv.org/pdf/2008.06423.pdf

General matching quantiles M-estimation

https://www.sciencedirect.com/science/article/abs/pii/S0167947320300323

"""
import torch
from torch.nn import MSELoss
import numpy as np
from depth.multivariate import *


class RegModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Linear(in_features=2, out_features=2)

    def forward(self, x):
        return self.model(x)


import numpy as np
import pandas as pd
import torch.distributions
from copulas.multivariate import GaussianMultivariate
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    N = 1000
    A = torch.tensor([[0.2, -0.5], [-0.5, 2.0]])
    cov = torch.matmul(A, A.T)
    mean = torch.tensor([-2.0, 2])
    # Y = torch.distributions.MultivariateNormal(loc=mean,
    #                                            covariance_matrix=cov).sample(torch.Size([N]))

    X = torch.distributions.MultivariateNormal(loc=torch.tensor([0.0, 0.0]), covariance_matrix=torch.eye(2)).sample(
        torch.Size([N]))
    ref1 = torch.distributions.MultivariateNormal(loc=torch.tensor([1.0, 1.0]), covariance_matrix=torch.eye(2)).sample(
        torch.Size([N]))
    ref2 = torch.distributions.MultivariateNormal(loc=torch.tensor([-1.0, -1.0]), covariance_matrix=torch.eye(2)).sample(
        torch.Size([N]))
    Y = torch.einsum('ij,bj->bi', A, X) + torch.tensor([-2, 2])
    m = torch.mean(Y, dim=0)
    c = torch.cov(Y.T)

    # copula_Y = GaussianMultivariate()
    # copula_X = GaussianMultivariate()

    # print('fitting copula')
    # copula_Y.fit(pd.DataFrame(Y.detach().numpy()))
    # copula_X.fit(pd.DataFrame(X.detach().numpy()))

    print(f'calculating cdf')
    # Y_cdf = copula_Y.cdf(Y.detach().numpy())
    # X_cdf = copula_X.cdf(X.detach().numpy())
    Y_depth = L2(Y.detach().numpy(), X.detach().numpy())
    X_depth = L2(X.detach().numpy(), X.detach().numpy())

    # min_cdf_val = max(min(Y_cdf), min(X_cdf))
    # max_cdf_val = min(max(Y_cdf), max(X_cdf))
    # # cdf matching
    print(f'copula matching')
    # slow, fix later
    # Y_idx_list = []
    # X_idx_list = []

    # for i in range(N):  # i to loop over Y_cdf
    #     if min_cdf_val <= Y_cdf[i] <= max_cdf_val:
    #         min_diff = 1000000
    #         i_matching_idx = -1
    #         for j in range(N):  # j to loop over X_cdf
    #             if min_cdf_val <= X_cdf[j] <= max_cdf_val:
    #                 diff = np.abs(X_cdf[j] - Y_cdf[i])
    #                 if diff < min_diff:
    #                     min_diff = diff
    #                     i_matching_idx = j
    #         if i_matching_idx > -1:
    #             Y_idx_list.append(i)
    #             X_idx_list.append(i_matching_idx)
    # sanity check
    # cdf_diff = np.abs(Y_cdf[Y_idx_list] - X_cdf[X_idx_list])
    # min_cdf_diff = np.min(cdf_diff)
    # max_cdf_diff = np.max(cdf_diff)
    # avg_cdf_diff = np.mean(cdf_diff)
    X_depth_sort_idx = np.argsort(X_depth)
    Y_depth_sort_idx = np.argsort(Y_depth)
    print(f'copula quantile matching')
    Y_depth_ordered = Y[Y_depth_sort_idx, :]
    X_depth_ordered = X[X_depth_sort_idx, :]

    # sanity check again
    # c1 = copula_Y.cdf(Y_q_ordered)
    # c2 = copula_X.cdf(X_q_ordered)
    # diff2 = np.abs(c1 - c2)
    # create LinReg
    y = Y_depth_ordered[:, 0]
    reg = LinearRegression(fit_intercept=False).fit(X_depth_ordered, Y_depth_ordered)
    s = reg.score(X_depth_ordered, Y_depth_ordered)
    model = RegModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for i in range(10000):
        optimizer.zero_grad()
        y_hat = model(torch.tensor(X_depth_ordered))
        loss = MSELoss()(y_hat, torch.tensor(Y_depth_ordered))
        print(loss.item())
        loss.backward()
        optimizer.step()
    print(f'finished')
