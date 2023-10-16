"""

Refs
Multivariate Order Statistics:
Theory and Application
Grant B. Weller1 and William F. Eddy
https://www.annualreviews.org/doi/pdf/10.1146/annurev-statistics-062614-042835

From geometric quantiles to halfspace depths:
A geometric approach for extremal behaviour.
https://arxiv.org/pdf/2306.10789.pdf

data-depth libray


QUANTILE TOMOGRAPHY: USING QUANTILES WITH
MULTIVARIATE DATA
https://www.researchgate.net/publication/1921805_Quantile_tomography_Using_quantiles_with_multivariate_data


DIRECTIONAL QUANTILE CLASSIFIERS
https://arxiv.org/pdf/2009.05007.pdf


https://arxiv.org/pdf/2004.01927.pdf

Quick Conclusion: Might work?


TODO :
apply regression for argsort data based on depths and compare against random

"""
import torch.distributions
import numpy as np
from depth.multivariate import *
from torch.nn import MSELoss
from sklearn.metrics import ndcg_score


def get_direction_vector(center: torch.Tensor, x: torch.Tensor):
    v = x - center
    normv = torch.norm(v)
    v_norm = v / normv
    test_ = torch.norm(v_norm)
    return v_norm


class LinRegModel(torch.nn.Module):
    def __init__(self, in_out_dim: int, hidden_dim: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential(torch.nn.Linear(in_out_dim, hidden_dim), torch.nn.Sigmoid(),
                                         torch.nn.Linear(hidden_dim, in_out_dim))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    D = 2
    N = 1000
    mio = torch.tensor([2.0] * D)
    Sigma = torch.diag(torch.tensor([9.0] * D))
    A = torch.sqrt(Sigma)
    base_distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=torch.eye(D))
    target_distribution = torch.distributions.MultivariateNormal(loc=mio, covariance_matrix=Sigma)
    X = base_distribution.sample(torch.Size([N]))
    # Y = torch.einsum('ij,bj->bi', A, X) + mio
    Y = torch.cos(X) + torch.exp(-X) + torch.tanh(X)
    # Y = target_distribution.sample(torch.Size([N]))
    # X_norm = torch.nn.functional.normalize(X, dim=1)
    # dummy = torch.norm(X_norm, dim=1)
    # Y_norm = torch.nn.functional.normalize(Y, dim=1)
    # dummy = torch.norm(Y_norm, dim=1)  # should all be 1's
    X_center = torch.mean(X,dim=0)
    Y_center = torch.mean(X,dim=0)
    Xv = torch.vmap(lambda x:get_direction_vector(X_center,x))(X)
    Yv = torch.vmap(lambda y: get_direction_vector(Y_center, y))(Y)
    X_L2_depth = L2(x=X.detach().numpy(), data=X.detach().numpy())
    Y_L2_depth = L2(x=Y.detach().numpy(), data=Y.detach().numpy())
    X_ranks = torch.argsort(torch.tensor(X_L2_depth))
    Y_ranks = torch.argsort(torch.tensor(Y_L2_depth))
    # # diff = X_L2_depth - Y_L2_depth
    # # depth_loss = np.mean(np.linalg.norm(X_L2_depth - Y_L2_depth))
    ndcg_test = ndcg_score(y_true=[list(np.arange(100, 0, -1))], y_score=[list(np.arange(0, 100, 1))])
    ndcg_ = ndcg_score(y_true=[X_L2_depth], y_score=[Y_L2_depth])

    print(f'ndcg = {ndcg_}')
    # # assert depth_loss <= 1e-4, f"{depth_loss}"
    # X_ordered = X[X_ranks, :]
    # Y_ordered = Y[Y_ranks, :]
    # dummyX = L2(x=X_ordered.detach().numpy(), data=X_ordered.detach().numpy())
    # dummyY = L2(x=Y_ordered.detach().numpy(), data=Y_ordered.detach().numpy())
    learning_rate = 0.01
    model = LinRegModel(in_out_dim=D)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(100000):
        optimizer.zero_grad()
        Y_hat = model(X)
        loss = MSELoss()(Y, Y_hat)
        print(f'loss = {loss.item()}')
        loss.backward()
        optimizer.step()
    print('finished')
