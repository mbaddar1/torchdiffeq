import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error
from torch.nn import MSELoss
import numpy as np
from sklearn.decomposition import PCA

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
            self.model = torch.nn.Sequential(torch.nn.Linear(in_out_dim, hidden_dim), torch.nn.Sigmoid(),
                                             torch.nn.Linear(hidden_dim, in_out_dim))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    D = 4
    # N = 10000
    batch_size = 10000
    niter = 1100
    q = torch.tensor(list(np.arange(0.1, 1, 0.001)), dtype=torch.float32)
    base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=torch.eye(D))
    mio = 10.0
    sigma = 5.0
    # A = torch.tensor([[0.1, -0.2, 0.1, -0.4], [0.1, 0.2, 0.3, 2.4], [0.1, 1.2, 2.3, 0.9], [0.2, 0.2, 0.1, 0.4]])
    # cov = torch.matmul(A.T, A)
    cov = torch.diag(torch.tensor([5.0, 10, 20, 40]))
    target_dist = torch.distributions.MultivariateNormal(loc=torch.tensor([mio] * D), covariance_matrix=cov)
    # X = base_dist.sample(torch.Size([N]))
    # Y = target_dist.sample(torch.Size([N]))
    # test_cov = torch.cov(Y.T)

    # Xq = torch.quantile(input=X, q=q, dim=0)
    # Yq = torch.quantile(input=Y, q=q, dim=0)
    # XqCov = torch.cov(Xq.T)
    # YqCov = torch.cov(Yq.T)
    # Ypca = torch.tensor(PCA().fit_transform(Y.detach().numpy()))
    # test_cov_pca = torch.cov(Ypca.T)
    # YqPca = torch.quantile(input=Ypca, q=q, dim=0)
    #
    learning_rate = 0.1
    model = Reg(in_out_dim=D, hidden_dim=50, type='linear')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(niter):
        # TODO train on batches from base and target distribution
        X = base_dist.sample(torch.Size([batch_size]))
        Y = target_dist.sample(torch.Size([batch_size]))
        # test_cov = torch.cov(Y.T)
        Xq = torch.quantile(input=X, q=q, dim=0)
        Yq = torch.quantile(input=Y, q=q, dim=0)
        if i > 1000:
            k = 10
        optimizer.zero_grad()
        Yq_hat = model(Xq)
        loss = MSELoss()(Yq, Yq_hat)
        print(f'i = {i} , loss = {loss}')

        # quick testing code
        if i > 1000:
            base_dist_scipy = multivariate_normal(mean=np.zeros(D), cov=np.eye(D))
            X_cdf = base_dist_scipy.cdf(x=Xq.detach().numpy())
            target_dist_scipy = multivariate_normal(mean=target_dist.mean.detach().numpy(),
                                                    cov=target_dist.covariance_matrix.detach().numpy())
            Y_pred_cdfs = target_dist_scipy.cdf(Yq_hat.detach().numpy())
            compare_dfs = pd.DataFrame({'Xcdf': X_cdf, 'Ycdf': Y_pred_cdfs})
            cdf_loss = mean_squared_error(y_true=X_cdf, y_pred=Y_pred_cdfs)
            print(f'>>> cdf_loss = {cdf_loss}')
        # update model
        loss.backward()
        optimizer.step()

    print(f'Model weight = {model.model.weight}')
    # X_test = base_dist.sample(torch.Size([N]))
    # Xq_test = torch.quantile(input=X_test, q=q, dim=0)
    #
    # X_cdf = base_dist_scipy.cdf(x=Xq.detach().numpy())
    # X_test_cdf = base_dist_scipy.cdf(x=Xq_test.detach().numpy())
    #
    # Y_pred = model(Xq)
    # Yq_pred = torch.quantile(input=Y_pred, q=q, dim=0)
    # target_dist_scipy = multivariate_normal(mean=target_dist.mean.detach().numpy(),
    #                                         cov=target_dist.covariance_matrix.detach().numpy())
    # Y_cdfs = target_dist_scipy.cdf(Yq.detach().numpy())
    # Y_pred_cdfs = target_dist_scipy.cdf(Yq_pred.detach().numpy())
    print('finished')
