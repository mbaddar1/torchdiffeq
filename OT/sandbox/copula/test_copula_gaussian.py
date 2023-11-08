import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from copulas.bivariate import Bivariate, CopulaTypes
from copulas.multivariate import GaussianMultivariate, VineCopula
from copulas.multivariate.tree import CenterTree
from copulas.visualization import scatter_2d, compare_2d


def generate_gaussian_mixture_samples(N: int, p: List[float]):
    assert abs(sum(p) - 1) <= 1e-6
    gaussians = []
    m1 = torch.tensor([-10.0, -8])
    A1 = torch.tensor([[-.3, 0.4], [0.2, -0.9]])
    C1 = torch.matmul(A1, A1.T)

    m2 = torch.tensor([8, 10.0])
    A2 = torch.tensor([[.3, -0.1], [-0.9, 0.4]])
    C2 = torch.matmul(A2, A2.T)

    # m3 = torch.tensor([-0.5, 0.1])
    # A3 = torch.tensor([[-.9, 8], [-0.4, 0.1]])
    # C3 = torch.matmul(A3, A3.T)

    gaussians.append(torch.distributions.MultivariateNormal(m1, C1))
    gaussians.append(torch.distributions.MultivariateNormal(m2, C2))
    # gaussians.append(torch.distributions.MultivariateNormal(m3, C3))

    c = np.random.choice(a=[0, 1], size=N, replace=True, p=p)
    samples = []
    for i in range(N):
        sample = gaussians[c[i]].sample(torch.Size([1]))
        samples.append(sample)
    return torch.cat(samples, dim=0)


if __name__ == '__main__':
    N = 2000
    mvn_mean = torch.tensor([-2.0, 2.0])
    A = torch.tensor([[-.3, 5], [0.2, -0.4]])
    mvn_cov = torch.matmul(A, A.T)
    mvn_dist = torch.distributions.MultivariateNormal(loc=mvn_mean, covariance_matrix=mvn_cov)
    # X = generate_gaussian_mixture_samples(N=N, p=[0.5, 0.5])
    X = mvn_dist.sample(torch.Size([N]))
    scatter_2d(data=pd.DataFrame(X.detach().numpy()))
    plt.savefig('X_mvn.jpg')
    #
    # TODO Debug Gaussian Copula Code - focus on univariate approx.
    copula = GaussianMultivariate()

    # copula=CenterTree()
    # copula = Bivariate(copula_type=CopulaTypes.GUMBEL)
    # copula = VineCopula(vine_type="regular")
    print(f'fitting')
    copula.fit(pd.DataFrame(X.detach().numpy()))

    print(f'sampling')
    X_copula = copula.sample(N)

    plt.clf()
    compare_2d(pd.DataFrame(X.detach().numpy()), pd.DataFrame(X_copula))
    plt.savefig('mvn_copula_compare.jpg')
