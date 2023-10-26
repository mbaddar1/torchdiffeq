"""
Script to play around with wasserstein distance and Optimal Transports
"""
import numpy as np
import torch.distributions
from torch.nn import MSELoss
import pingouin as pg
from itertools import product
import numpy as np
from scipy.stats import multivariate_normal

EPS = 1e-2


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class nn(torch.nn.Module):
    def __init__(self, inputSize, outputSize, hidden_dim=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = torch.nn.Sequential(torch.nn.Linear(inputSize, hidden_dim),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_dim, hidden_dim),
                                     torch.nn.Tanh(),
                                     torch.nn.Linear(hidden_dim, outputSize))

    def forward(self, x):
        out = self.m(x)
        return out


if __name__ == '__main__':
    N = 1000
    n_iter = 500
    # linearRegression(inputSize=1, outputSize=1)

    losses = []
    freq = 1
    loc = [0.1, -0.1, 0.2, 0.9]
    D = len(loc)
    Sigma = torch.diag(torch.tensor([0.25, 0.25, 0.25, 0.25]))
    A = torch.diag(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.norm(torch.matmul(A, A.T) - Sigma) <= EPS
    #
    model = nn(inputSize=D, outputSize=D)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with_transformation = True
    with_perm = False
    base_distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=torch.eye(D))
    target_distribution = torch.distributions.MultivariateNormal(loc=torch.tensor(loc), covariance_matrix=Sigma)
    target_distribution_scipy = multivariate_normal(mean=target_distribution.mean.detach().numpy(),
                                                    cov=target_distribution.covariance_matrix.detach().numpy())
    x = base_distribution.sample(torch.Size([N]))
    # https://saturncloud.io/blog/multivariate-normal-cdf-in-python-using-scipy/
    base_distribution_scipy = multivariate_normal(mean=np.zeros(D), cov=np.eye(D))
    cdf_base_samples = torch.tensor(list(map(lambda g: base_distribution_scipy.cdf(g), x.tolist())))

    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    if with_transformation:
        y = torch.einsum('ij,bj->bi', A, x) + torch.tensor(loc)
        cdf_target_samples = torch.tensor(list(map(lambda g: target_distribution_scipy.cdf(g), y.tolist())))
        cdfs_mse_loss = MSELoss()(cdf_target_samples, cdf_base_samples)
        res = pg.multivariate_normality(X=y.detach().numpy(), alpha=0.01)
        if not res.normal:
            print(f'Warning : HZ test failed with p-value = {res.pval}')

        sample_mean = torch.mean(y, dim=0)
        sample_cov = torch.cov(y.T)
        sample_mean_mse_loss = MSELoss()(sample_mean, torch.tensor(loc))
        sample_cov_mse_loss = MSELoss()(sample_cov, Sigma)
        if sample_mean_mse_loss > EPS:
            print(f"Warning : sample_mean_mse_loss = {sample_mean}")
        if sample_cov_mse_loss > EPS:
            print(f"Warning : sample_cov_mse_loss = {sample_cov_mse_loss}")
        if with_perm:
            x = x[torch.randperm(N)]
    else:
        y = target_distribution.sample(torch.Size([N]))

    x_list = x.tolist()
    y_list = y.tolist()
    xyprod = list(product(x_list, y_list))
    x_all = torch.tensor(list(map(lambda x: x[0], xyprod)))
    y_all = torch.tensor(list(map(lambda x: x[1], xyprod)))
    for i in range(n_iter):
        # idx = np.random.choice(a=list(np.arange(0, N * N)), size=N)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = MSELoss()(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if i > freq and i % freq == 0:
            print(f"itr = {i} = {np.nanmean(losses[:-freq])}")

    y_hat_test = model(x)
    u = torch.sort(torch.norm(y_hat_test - y, dim=1))
    hz = pg.multivariate_normality(y_hat_test.detach().numpy())
    print(hz)
    print("sample mean")
    print(torch.mean(y_hat_test, dim=0))
    print("sample cov")
    print(torch.cov(y_hat_test.T))
    print("MSE mean")
    print(MSELoss()(torch.mean(y_hat_test, dim=0), torch.tensor(loc)).item())
    print("MSE Cov")
    print(MSELoss()(torch.cov(y_hat_test.T), Sigma).item())
