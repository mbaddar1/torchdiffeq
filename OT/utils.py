import logging
from typing import List

import torch
from torch.linalg import multi_dot
from torch.nn import MSELoss

from GMSOC.functional_tt_fabrique import orthpoly, Extended_TensorTrain
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


def run_tt_als(x: torch.Tensor, y: torch.Tensor, ETT_fits: List[Extended_TensorTrain], test_ratio: float,
               tol: float, domain_stripe: List[float]) -> None:
    """

    :param ETT_fits:
    :param domain_stripe:
    :param x: pure x, no time

    :param y:
    :param test_ratio:
    :param tol:
    :return:
    """
    logger = logging.getLogger()
    N = x.shape[0]
    # N_test = int(test_ratio * N)
    # N_train = N - N_test
    # x_device = x.get_device() if x.is_cuda else torch.device('cpu')
    # x_aug = torch.cat(tensors=[x, torch.tensor([t]).repeat(N, 1).to(x_device)], dim=1)
    Dx = x.shape[1]
    Dy = y.shape[1]
    #
    # degrees = [poly_degree] * Dx_aug
    # ranks = [1] + [rank] * (Dx_aug - 1) + [1]
    # order = len(degrees)  # not used, but for debugging only
    # domain = [domain_stripe for _ in range(Dx_aug)]
    # op = orthopoly(degrees, domain, device=x_device)
    x_aug_domain_adjusted = domain_adjust(x=x, domain_stripe=domain_stripe)
    is_domain_adjusted(x=x_aug_domain_adjusted, domain_stripe=domain_stripe)
    assert len(ETT_fits) == Dy, "len(ETT_fits) must be equal to Dy"
    for i, ETT_fit in enumerate(ETT_fits):
        assert isinstance(ETT_fit, Extended_TensorTrain), (f"ETT_fit {i} must be of type "
                                                           f"{Extended_TensorTrain.__class__.__name__},"
                                                           f"found {type(ETT_fit)} ")
    # ETT_fits = [Extended_TensorTrain(op, ranks, device=x_device) for _ in range(Dy)]
    y_predicted_list = []

    for j in range(Dy):
        y_d = y[:, j].view(-1, 1)
        # ALS parameters
        reg_coeff = 1e-2
        iterations = 4
        rule = None
        # rule = tt.Dörfler_Adaptivity(delta = 1e-6,  maxranks = [32]*(n-1), dims = [feature_dim]*n, rankincr = 1)
        ETT_fits[j].fit(x=x_aug_domain_adjusted.type(torch.float64)[:N, :],
                        y=y_d.type(torch.float64)[:N, :],
                        iterations=iterations, rule=rule, tol=tol,
                        verboselevel=1, reg_param=reg_coeff)
    #     ETT_fits[j].tt.set_core(Dx - 1)
    #     # train_error = (torch.norm(ETT_fits[j](x_domain_adjusted.type(torch.float64)[:N_train, :]) -
    #     #                           y_d.type(torch.float64)[:N_train, :]) ** 2 / torch.norm(
    #     #     y_d.type(torch.float64)[:N_train, :]) ** 2).item()
    #     #
    #     # val_error = (torch.norm(ETT_fits[j](x_domain_adjusted.type(torch.float64)[N_train:, :]) -
    #     #                         y_d.type(torch.float64)[N_train:, :]) ** 2 / torch.norm(
    #     #     y_d.type(torch.float64)[N_train:, :]) ** 2).item()
    #     y_d_predict_train = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64)[:N, :])
    #     y_d_predict_val = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64)[N:, :])
    #
    #     train_rmse = torch.sqrt(MSELoss()(y_d_predict_train, y_d.type(torch.float64)[:N, :]))
    #     test_rmse = torch.sqrt(MSELoss()(y_d_predict_val, y_d.type(torch.float64)[N:, :]))
    #     logger.info(f'For j ( d-th dim of y)  = {j} :TT-ALS RMSE on training set = {train_rmse}')
    #     logger.info(f'For j (d-th dim of y)  = {j}: TT-ALS RMSE on test set = {test_rmse}')
    #     #
    #     y_d_predicted = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64))
    #     y_predicted_list.append(y_d_predicted)
    #     logger.info("======== Finished TT-ALS training ============")
    # y_predicted = torch.cat(tensors=y_predicted_list, dim=1)
    # prediction_tot_rmse = torch.sqrt(MSELoss()(y, y_predicted))
    # return {'prediction_tot_rmse': prediction_tot_rmse}


def is_domain_adjusted(x: torch.Tensor, domain_stripe: List[float], eps: float = 0.05) -> bool:
    """
    :param domain_stripe:
    :param eps:
    :param x:
    :return:
    """
    D = x.shape[1]
    dtype = torch.float32
    x_min = torch.min(input=x, dim=0).values.to(torch.device("cpu"))
    x_max = torch.max(input=x, dim=0).values.to(torch.device("cpu"))
    x_min_ref = torch.tensor(domain_stripe[0]).repeat(D - 1)  # skip time dim, as it is passed as single value
    x_max_ref = torch.tensor(domain_stripe[1]).repeat(D - 1)
    abs_x_max_diff = torch.abs(x_max.detach().type(dtype)[:D - 1] - x_max_ref.type(dtype))
    abs_x_min_diff = torch.abs(x_min.detach().type(dtype)[:D - 1] - x_min_ref.type(dtype))
    max_of_max_diff = torch.max(abs_x_max_diff)
    max_of_min_diff = torch.max(abs_x_min_diff)
    if max_of_max_diff > eps:
        return False
    if max_of_min_diff > eps:
        return False
    # time dimension
    t_max = torch.max(x[:, D - 1], dim=0).values.to(torch.device("cpu")).item()
    t_min = torch.min(x[:, D - 1], dim=0).values.to(torch.device("cpu")).item()
    if abs(t_max - t_min) < 1e-6:  # almost_equal
        # because of values are the same, it will be clamped to the lowest domain stripe value
        if abs(t_max - domain_stripe[0]) > eps:
            return False
    return True


def domain_adjust(x: torch.Tensor, domain_stripe: List[float]) -> torch.Tensor:
    """

    :param x:
    :param domain_stripe:
    :return:
    """
    epsilon = 0.01
    x_min = torch.min(x, dim=0)
    x_max = torch.max(x, dim=0)
    x_scaled = (x - x_min.values) / (x_max.values - x_min.values + epsilon)
    # FIXME, for debugging
    x_scaled_min = torch.min(x_scaled, dim=0)
    x_scaled_max = torch.max(x_scaled, dim=0)
    x_domain = domain_stripe[0] + (domain_stripe[1] - domain_stripe[0]) * x_scaled
    # FIXME for debugging
    x_domain_min = torch.min(x_domain, dim=0)
    x_domain_max = torch.max(x_domain, dim=0)
    return x_domain


def get_ETTs(D_in: int, D_out: int, rank: int, domain_stripe: List[float], poly_degree: int, device: torch.device):
    """

    :param D_in:
    :param D_out:
    :param rank:
    :param domain_stripe:
    :param poly_degree:
    :param device:
    :return:
    """
    degrees = [poly_degree] * D_in
    ranks = [1] + [rank] * (D_in - 1) + [1]
    # order = len(degrees)  # not used, but for debugging only
    domain = [domain_stripe for _ in range(D_in)]
    op = orthpoly(degrees, domain, device)
    ETT_fits = [Extended_TensorTrain(op, ranks, device) for _ in range(D_out)]
    return ETT_fits


def ETT_fits_predict(ETT_fits: List[Extended_TensorTrain], x: torch.Tensor,
                     domain_stripe: List[float]) -> torch.Tensor:
    # N = x.size()[0]
    Dx = x.size()[1]
    # if x.is_cuda:
    #     x_device = x.get_device()
    # else:
    #     x_device = torch.device('cpu')

    x_domain_adjusted = domain_adjust(x=x, domain_stripe=domain_stripe)
    assert is_domain_adjusted(x=x_domain_adjusted, domain_stripe=domain_stripe)
    assert len(ETT_fits) == Dx

    yd_hat_list = []

    for d in range(Dx):
        assert len(ETT_fits[d].tt.dims) == Dx, (f"ETT order (num of degrees)  must = Dz_aug : "
                                                    f"{len(ETT_fits[d].tt.dims)} != {Dx}")
        ETT_fits[d].tt.set_core(Dx - 1)
        yd_hat = ETT_fits[d](x_domain_adjusted)
        yd_hat_list.append(yd_hat)
    y_hat = torch.cat(tensors=yd_hat_list, dim=1)
    return y_hat

