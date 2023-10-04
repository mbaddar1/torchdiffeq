import logging
from sqrtm import sqrtm

import torch
from torch.linalg import multi_dot
from pingouin import multivariate_normality

#
logger = logging.getLogger()


def is_matrix_positive_definite(M: torch.Tensor, semi: bool) -> bool:
    """
    https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf
    https://en.wikipedia.org/wiki/Definite_matrix
    :param semi:
    :param M:
    :return:
    """
    # check symmetry
    if not torch.equal(M, M.T):
        return False
    eig_vals = torch.linalg.eigvals(M)
    if semi:
        return torch.all(torch.vmap(lambda x: x >= 0.0)(torch.real(eig_vals))).item()
    else:
        return torch.all(torch.vmap(lambda x: x > 0.0)(torch.real(eig_vals))).item()


def wasserstein_distance_two_gaussians(m1: torch.Tensor, m2: torch.Tensor, C1: torch.Tensor,
                                       C2: torch.Tensor) -> float:
    """
    https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
    https://en.wikipedia.org/wiki/Wasserstein_metric
    https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

    :param m1:
    :param m2:
    :param C1:
    :param C2:
    :return:
    """
    # check shapes
    assert m1.dim() == 1
    assert m2.dim() == 1
    assert m1.size()[0] == m2.size()[0]
    D = m1.size()[0]
    assert C1.dim() == 2
    assert C2.dim() == 2
    assert C1.size()[0] == C1.size()[1] and C2.size()[0] == C2.size()[1]
    assert C1.size()[0] == C2.size()[0]
    assert C1.size()[0] == D
    # https://pytorch.org/docs/stable/generated/torch.norm.html
    # check positive definite matrix
    assert is_matrix_positive_definite(M=C1, semi=True)
    assert is_matrix_positive_definite(M=C2, semi=True)

    K = multi_dot(tensors=[sqrtm(C2), C1, sqrtm(C2)])
    d = torch.pow(torch.norm(m1 - m2), 2) + torch.trace(C1 + C2 - 2 * sqrtm(K))
    return d.item()


if __name__ == '__main__':
    mio1 = torch.tensor([0.1, 0.2, 0.3])
    mio2 = torch.tensor([-0.1, -0.2, -0.3])
    cov1 = torch.tensor([[0.9, 0, 0], [0, 0.7, 0], [0, 0, 0.1]])
    cov2 = torch.tensor([[0.3, 0, 0], [0, 0.1, 0], [0, 0, 0.9]])

    # assert wasserstein_distance_two_gaussians(m1=mio1, m2=mio1, C1=cov1,
    #                                           C2=cov1) < wasserstein_distance_two_gaussians(m1=mio1, m2=mio1 + 0.01,
    #                                                                                         C1=cov1, C2=cov1)
    # assert wasserstein_distance_two_gaussians(m1=mio1, m2=mio1, C1=cov1,
    #                                           C2=cov1) < wasserstein_distance_two_gaussians(m1=mio1, m2=mio1,
    #                                                                                         C1=cov1, C2=1.2 * cov1)
    N = 10000
    mvn1 = torch.distributions.MultivariateNormal(loc=mio1, covariance_matrix=cov1)
    mvn2 = torch.distributions.MultivariateNormal(loc=mio2, covariance_matrix=cov2)

    z_sample1 = mvn1.sample(torch.Size([10000]))
    z_sample2 = mvn2.sample(torch.Size([10000]))

    z_sample_mean1 = torch.mean(z_sample1, dim=0)
    z_sample_mean2 = torch.mean(z_sample2, dim=0)

    z_sample_cov1 = torch.cov(z_sample1.T)
    z_sample_cov2 = torch.cov(z_sample2.T)

    wd_sample1 = wasserstein_distance_two_gaussians(m1=mio1, m2=z_sample_mean1, C1=cov1, C2=z_sample_cov1)
    wd_sample2 = wasserstein_distance_two_gaussians(m1=mio1, m2=z_sample_mean2, C1=cov1, C2=z_sample_cov2)
    assert wd_sample1 < wd_sample2

    # hz_res = multivariate_normality(X=z_sample.detach().numpy())
    logpz11 = mvn1.log_prob(mio1)
    logpz21 = mvn2.log_prob(z_sample1)
    logpz11_mean = torch.mean(logpz11)
    logpz21_mean = torch.mean(logpz21)
    print('Finished')
    # """
    # Test using a distribution with given params and sampled params
    # """
