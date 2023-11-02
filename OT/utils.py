import logging
from typing import List, Union, Iterable

import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA, IncrementalPCA
from statsmodels.distributions import ECDF
from torch.linalg import multi_dot
from torch.nn import MSELoss
import pingouin as pg
from GMSOC.functional_tt_fabrique import orthpoly, Extended_TensorTrain
from OT.sqrtm import sqrtm
from scipy.stats import norm


def get_base_dist_quantiles(p_levels: torch.Tensor, base_dist: torch.distributions.Distribution,
                            torch_dtype: torch.dtype, torch_device: torch.device) -> torch.Tensor:
    if isinstance(base_dist, torch.distributions.MultivariateNormal):
        D = base_dist.mean.size()[0]
        diag_cov = torch.diagonal(base_dist.covariance_matrix)
        # assert cov matrix is diagonal
        tmp_cov = torch.diag(diag_cov)
        err = torch.linalg.norm(tmp_cov - base_dist.covariance_matrix)
        assert err <= 1e-6, "base dist cov is not diagonal"
        quantiles = torch.stack(
            [torch.tensor(norm.ppf(q=p_levels.detach().numpy(), loc=base_dist.mean[d].item(), scale=diag_cov[d].item()))
             for d
             in range(D)], dim=1).type(torch_dtype).to(torch_device)
        return quantiles
    else:
        raise ValueError(f'Unsupported base-dist = {type(base_dist)}')


def uv_sample(Yq: torch.Tensor, N: int, u_levels: torch.Tensor, u_step: float, interp: str) -> torch.Tensor:
    """
    There is some kind of redundancy in parameters. It's intentional and try to alleviate it by some assertions
    :param interp:
    :param u_step:
    :param u_levels:
    :param Yq:
    :param N:
    :return:
    """
    # interpolation methods
    interp_methods = ["linear", "cubic"]
    # some assertions for params
    assert interp in interp_methods, f"interp param must be one of {interp_methods}"
    eps = 1e-6
    Yq_size = Yq.size()
    assert len(Yq_size) == 1, "Yq must be of 1 dim"
    u_levels_size = u_levels.size()
    assert (len(u_levels.size()) == 1), "u_levels must be of dim 1"
    assert u_levels_size[0] == Yq_size[0], "Yq must have same 1-dim size as u_levels"
    assert np.abs(u_levels[0] - 0) <= eps, "u_levels[0] must be 0"
    assert 0.99 <= u_levels[-1] < 1, "u_levels[-1] must be >= 0.99 and < 1"
    assert np.abs(u_step - 1.0 / (u_levels_size[0] - 1)) <= eps, "u_levels size must compatible with u_step"
    #
    u_sample = torch.distributions.Uniform(0, 1).sample(torch.Size([N]))
    Y_sample = None
    if interp == 'linear':
        idx_low = torch.floor(u_sample / u_step).type(torch.int)
        idx_high = idx_low + 1
        u_low = u_levels[idx_low]
        u_high = u_levels[idx_high]
        Yq_low = Yq[idx_low]
        Yq_high = Yq[idx_high]
        m = (u_sample - u_low) / (u_high - u_low)
        Y_sample = torch.mul(m, Yq_high - Yq_low) + Yq_low
    elif interp == 'cubic':
        cs = CubicSpline(x=u_levels.detach().numpy(), y=Yq.detach().numpy())
        Y_sample_np = cs(u_sample.detach().numpy())
        Y_sample = torch.tensor(Y_sample_np, dtype=torch.float32)
    else:
        raise ValueError(f'Invalid interp param val = {interp} : must be one of {interp_methods}')
    return Y_sample


def check_wd(wd: float, wd_thresh: float):
    if wd <= wd_thresh:
        print(f"wd is small enough : {wd} <= {wd_thresh}")
    else:
        print(f'wd is large  = {wd} >= {wd_thresh}')


def validate_qq_model(base_dist: torch.distributions.Distribution,
                      target_dist: torch.distributions.Distribution,
                      model: Union[torch.nn.Module, List[Extended_TensorTrain]], N: int,
                      p_levels: torch.Tensor, p_step: float, train_transformer: IncrementalPCA, repeats: int, D: int,
                      torch_dtype: torch.dtype, torch_device: torch.device) -> dict:
    mses_qq = []
    mse_cdfs = []
    wd_list = []
    mvn_hz_test_res_list = []
    for i in range(repeats):
        print(f'validation iteration {i + 1} out of {repeats}')
        # q-q validation
        X_test = base_dist.sample(torch.Size([N]))
        Xq_test = torch.quantile(input=X_test, q=p_levels.type(X_test.dtype), dim=0).type(torch_dtype).to(torch_device)
        # Xq_test_aug = torch.cat([Xq_test, u_levels.view(-1, 1)], dim=1)
        Y_test = target_dist.sample(torch.Size([N])).type(torch_dtype).to(torch_device)
        # ic => indep components
        Y_test_ic = torch.tensor(IncrementalPCA(whiten=train_transformer.whiten).fit_transform(Y_test.detach().numpy()),
                                 dtype=torch_dtype, device=torch_device)
        Yq_comp_test_ref = torch.quantile(input=Y_test_ic, dim=0, q=p_levels.type(Y_test_ic.dtype))
        if isinstance(model, torch.nn.Module):
            Yq_pred = model(Xq_test)
        elif isinstance(model, list) and all([isinstance(model[k], Extended_TensorTrain)] for k in range(len(model))):
            Yq_pred = ETT_fits_predict(ETT_fits=model, x=Xq_test, domain_stripe=[-1, 1])
        else:
            raise ValueError(f'Unsupported model type')
        mse = MSELoss()(Yq_comp_test_ref, Yq_pred).item()
        mses_qq.append(mse)
        # cdf validation
        for j in range(D):
            plt.plot(p_levels.detach().numpy(), Yq_comp_test_ref[:, j].detach().numpy())
            plt.plot(p_levels.detach().numpy(), Yq_pred[:, j].detach().numpy())
            plt.savefig(f'cdf_d_{j}.png')
            plt.clf()
            plt.plot(Yq_comp_test_ref[:, j].detach().numpy(), Yq_comp_test_ref[:, j].detach().numpy())
            plt.plot(Yq_comp_test_ref[:, j].detach().numpy(), Yq_pred[:, j].detach().numpy())
            plt.savefig(f'qq_d_{j}.png')
            plt.clf()
        mse_cdf_repeat = []
        for j in range(D):
            y_ica_j = Yq_comp_test_ref[:, j].detach().numpy()
            ecdf = ECDF(x=y_ica_j)
            cdf_ref = ecdf(Yq_comp_test_ref[:, j].detach().numpy())
            cdf_est = ecdf(Yq_pred[:, j].detach().numpy())
            plt.clf()
            mse_cdf_j = MSELoss()(torch.tensor(cdf_ref), torch.tensor(cdf_est))
            mse_cdf_repeat.append(mse_cdf_j)
        mse_cdfs.append(torch.tensor(mse_cdf_repeat))
        # validate the reconstruction process
        if isinstance(target_dist, torch.distributions.MultivariateNormal):
            Y_comp_qinv_sample = torch.stack([
                uv_sample(Yq=Yq_pred[:, i].reshape(-1), N=N, u_levels=p_levels, u_step=p_step, interp='cubic')
                for i in range(D)]).T.type(torch_dtype).to(torch_device)
            # FIXME remove : set of debug vars
            # Y_comp_qinv_sample = Y_comp_qinv_sample - torch.mean(Y_comp_qinv_sample, dim=0)  # fixme: quick fix

            Y_recons = torch.tensor(train_transformer.inverse_transform(Y_comp_qinv_sample.detach().numpy())).type(
                torch_dtype).to(torch_device)
            wd_baseline = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                             m2=torch.mean(Y_test, dim=0), C2=torch.cov(Y_test.T))
            mean_recons = torch.mean(Y_recons, dim=0)
            cov_recons = torch.cov(Y_recons.T)
            wd_recons = wasserstein_distance_two_gaussians(m1=target_dist.mean, C1=target_dist.covariance_matrix,
                                                           m2=mean_recons, C2=cov_recons)
            mvn_hz_baseline = pg.multivariate_normality(X=Y_test.detach().numpy())
            mvn_hz_recons = pg.multivariate_normality(X=Y_recons.detach().numpy())
            wd_list.append({'baseline': wd_baseline, 'reconstruct': wd_recons})
            mvn_hz_test_res_list.append({'baseline': mvn_hz_baseline, 'reconstruct': mvn_hz_recons})

    res = dict()
    # res['ica_qq_mses'] = mses_qq
    # res['cdf_mses'] = mse_cdfs
    res['wd'] = wd_list
    res['mvn_hz'] = mvn_hz_test_res_list
    return res


##############
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
               tol: float, domain_stripe: List[float], max_iter: int, regularization_coeff: float) -> None:
    """

    :param regularization_coeff:
    :param max_iter:
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
    x_domain_adjusted = domain_adjust(x=x, domain_stripe=domain_stripe)
    is_domain_adjusted(x=x_domain_adjusted, domain_stripe=domain_stripe)
    assert len(ETT_fits) == Dy, "len(ETT_fits) must be equal to Dy"
    for i, ETT_fit in enumerate(ETT_fits):
        assert isinstance(ETT_fit, Extended_TensorTrain), (f"ETT_fit {i} must be of type "
                                                           f"{Extended_TensorTrain.__class__.__name__},"
                                                           f"found {type(ETT_fit)} ")
    # ETT_fits = [Extended_TensorTrain(op, ranks, device=x_device) for _ in range(Dy)]
    # y_predicted_list = []

    for j in range(Dy):
        y_d = y[:, j].view(-1, 1)
        # ALS parameters
        rule = None
        print(f'*** Running TT-ALS for d = {j} ***')
        ETT_fits[j].fit(x=x_domain_adjusted.type(torch.float64)[:N, :],
                        y=y_d.type(torch.float64)[:N, :],
                        iterations=max_iter, rule=rule, tol=tol,
                        verboselevel=1, reg_param=regularization_coeff)
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
    for d, ETT in enumerate(ETT_fits):
        assert ETT.d == Dx

    yd_hat_list = []

    for d in range(len(ETT_fits)):
        assert len(ETT_fits[d].tt.dims) == Dx, (f"ETT order (num of degrees)  must = Dz_aug : "
                                                f"{len(ETT_fits[d].tt.dims)} != {Dx}")
        ETT_fits[d].tt.set_core(Dx - 1)
        yd_hat = ETT_fits[d](x_domain_adjusted)
        yd_hat_list.append(yd_hat)
    y_hat = torch.cat(tensors=yd_hat_list, dim=1)
    return y_hat
