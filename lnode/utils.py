from typing import List

import torch
from torch import dtype


def domain_adjust(x: torch.Tensor, domain: List[float]) -> torch.Tensor:
    """

    :param x:
    :param domain:
    :return:
    """
    epsilon = 0.01
    x_min = torch.min(x, dim=0)
    x_max = torch.max(x, dim=0)
    x_scaled = (x - x_min.values) / (x_max.values - x_min.values + epsilon)
    # FIXME, for debugging
    x_scaled_min = torch.min(x_scaled, dim=0)
    x_scaled_max = torch.max(x_scaled, dim=0)
    x_domain = domain[0] + (domain[1] - domain[0]) * x_scaled
    # FIXME for debugging
    x_domain_min = torch.min(x_domain, dim=0)
    x_domain_max = torch.max(x_domain, dim=0)
    return x_domain


def is_domain_adjusted(x: torch.Tensor, domain: List[float], eps: float = 0.05) -> bool:
    """
    :param eps:
    :param x:
    :param domain: 
    :return: 
    """
    D = x.shape[1]
    dtype = torch.float32
    x_min = torch.min(input=x, dim=0).values.to(torch.device("cpu"))
    x_max = torch.max(input=x, dim=0).values.to(torch.device("cpu"))
    x_min_ref = torch.tensor(domain[0]).repeat(D)
    x_max_ref = torch.tensor(domain[1]).repeat(D)
    abs_x_max_diff = torch.abs(x_max.detach().type(dtype) - x_max_ref.type(dtype))
    abs_x_min_diff = torch.abs(x_min.detach().type(dtype) - x_min_ref.type(dtype))
    max_of_max_diff = torch.max(abs_x_max_diff)
    max_of_min_diff = torch.max(abs_x_min_diff)
    if max_of_max_diff > eps:
        return False
    if max_of_min_diff > eps:
        return False
    return True
