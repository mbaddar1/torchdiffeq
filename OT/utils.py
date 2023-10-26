import torch
from torch.linalg import multi_dot

from OT.sqrtm import sqrtm


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
