import logging
from typing import List, Iterable

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Optimizer
import pingouin as pg
from GMSOC.tt_fabrique import TensorTrain
from examples.models import CNF
from GMSOC.functional_tt_fabrique import Extended_TensorTrain, orthpoly
from metrics import wasserstein_distance_two_gaussians

# Global variables

ODE_DIRECTIONS = {"forward": 1, "backward": -1}


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


def run_tt_als(x: torch.Tensor, t: float, y: torch.Tensor, ETT_fits: List[Extended_TensorTrain], test_ratio: float,
               tol: float, domain_stripe: List[float]) -> dict:
    """

    :param ETT_fits:
    :param domain_stripe:
    :param x: pure x, no time
    :param t:
    :param y:
    :param test_ratio:
    :param tol:
    :return:
    """
    logger = logging.getLogger()
    N = x.shape[0]
    N_test = int(test_ratio * N)
    N_train = N - N_test
    x_device = x.get_device() if x.is_cuda else torch.device('cpu')
    x_aug = torch.cat(tensors=[x, torch.tensor([t]).repeat(N, 1).to(x_device)], dim=1)
    Dx_aug = x_aug.shape[1]
    Dy = y.shape[1]
    #
    # degrees = [poly_degree] * Dx_aug
    # ranks = [1] + [rank] * (Dx_aug - 1) + [1]
    # order = len(degrees)  # not used, but for debugging only
    # domain = [domain_stripe for _ in range(Dx_aug)]
    # op = orthopoly(degrees, domain, device=x_device)
    x_aug_domain_adjusted = domain_adjust(x=x_aug, domain_stripe=domain_stripe)
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
        ETT_fits[j].fit(x=x_aug_domain_adjusted.type(torch.float64)[:N_train, :],
                        y=y_d.type(torch.float64)[:N_train, :],
                        iterations=iterations, rule=rule, tol=tol,
                        verboselevel=1, reg_param=reg_coeff)
        ETT_fits[j].tt.set_core(Dx_aug - 1)
        # train_error = (torch.norm(ETT_fits[j](x_domain_adjusted.type(torch.float64)[:N_train, :]) -
        #                           y_d.type(torch.float64)[:N_train, :]) ** 2 / torch.norm(
        #     y_d.type(torch.float64)[:N_train, :]) ** 2).item()
        #
        # val_error = (torch.norm(ETT_fits[j](x_domain_adjusted.type(torch.float64)[N_train:, :]) -
        #                         y_d.type(torch.float64)[N_train:, :]) ** 2 / torch.norm(
        #     y_d.type(torch.float64)[N_train:, :]) ** 2).item()
        y_d_predict_train = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64)[:N_train, :])
        y_d_predict_val = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64)[N_train:, :])

        train_rmse = torch.sqrt(MSELoss()(y_d_predict_train, y_d.type(torch.float64)[:N_train, :]))
        test_rmse = torch.sqrt(MSELoss()(y_d_predict_val, y_d.type(torch.float64)[N_train:, :]))
        logger.info(f'For j ( d-th dim of y)  = {j} :TT-ALS RMSE on training set = {train_rmse}')
        logger.info(f'For j (d-th dim of y)  = {j}: TT-ALS RMSE on test set = {test_rmse}')
        #
        y_d_predicted = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64))
        y_predicted_list.append(y_d_predicted)
        logger.info("======== Finished TT-ALS training ============")
    y_predicted = torch.cat(tensors=y_predicted_list, dim=1)
    prediction_tot_rmse = torch.sqrt(MSELoss()(y, y_predicted))
    return {'prediction_tot_rmse': prediction_tot_rmse}


def tt_rk_step(t: float, z: torch.Tensor, log_p_z: torch.Tensor, h: float, ETT_fits: List[Extended_TensorTrain],
               direction: str, domain_stripe: List[float]) -> dict:
    """

    :param log_p_z:
    :param z: z(t_n)
    :param direction:
    :param domain_stripe:
    :param t: t_n
    :param h:
    :param ETT_fits:
    :return:
    """
    f = ETT_fits_predict(ETT_fits=ETT_fits, z=z, t=t, domain_stripe=domain_stripe)
    trace_df_dz = ETT_fits_trace_grad(ETT_fits=ETT_fits, z=z, t=t, domain_stripe=domain_stripe)
    # dlog(p(z))/dt = -trace(df/dz)
    g = -trace_df_dz

    assert direction in ODE_DIRECTIONS.keys(), f"rk-step direction must be in {ODE_DIRECTIONS.keys()}"
    if direction == "forward":
        direction_num = 1
    elif direction == "backward":
        direction_num = -1
    else:
        raise ValueError(f"Unknown rk-step direction : {direction}")
    z_prime = z + direction_num * h * f  # z_prime = z(t_{n+1}) or z(t_{n-1}) , given the direction
    log_p_z_prime = log_p_z.view(-1, 1) + direction_num * h * g.view(-1, 1)

    # FIXME : Older attempt to use cnf main code trace(df/dz) code, but now using the ETT grad one
    #   To remove later when we make sure everything is OK
    # # d log p(z(t)) /dt = -tr(df / dz ) = g
    # # log p(z(tN-h)) = log p (z(tN)) - h * g
    # # FIXME RuntimeError: One of the differentiated Tensors does not require grad
    # #   https://discuss.pytorch.org/t/one-of-the-differentiated-tensors-does-not-require-grad/54694
    # f_with_grad = torch.nn.Parameter(data=f)
    # z_tN_with_grad = torch.nn.Parameter(data=z_tN)
    # # FIXME , this assertion should be inside the tr(df/dz) function ??
    # assert f_with_grad.requires_grad and z_tN_with_grad.requires_grad, \
    #     "before calculating -tr(df/dz) with torch.autograd, f and z must have requires_grad = True"
    #
    # g = -trace_df_dz_for_hybrid_trajectory(f=f_with_grad, z=z_tN_with_grad)
    # log_p_ztN_minus_h_pred = log_p_z_tN - h * g

    return {'dzdt': f, 'z_prime': z_prime, 'log_p_z_prime': log_p_z_prime}


def ETT_fits_predict(ETT_fits: List[Extended_TensorTrain], z: torch.Tensor, t: float,
                     domain_stripe: List[float]) -> torch.Tensor:
    N = z.size()[0]
    Dz = z.size()[1]
    if z.is_cuda:
        z_device = z.get_device()
    else:
        z_device = torch.device('cpu')
    z_aug = torch.cat(tensors=[z, torch.tensor(t).repeat(N, 1).to(z_device)], dim=1)
    Dz_aug = z_aug.size()[1]
    z_aug_domain_adjusted = domain_adjust(x=z_aug, domain_stripe=domain_stripe)
    assert is_domain_adjusted(x=z_aug_domain_adjusted, domain_stripe=domain_stripe)
    assert len(ETT_fits) == Dz
    # f = dz/dt
    fd_list = []

    for d in range(Dz):
        assert len(ETT_fits[d].tt.dims) == Dz_aug, (f"ETT order (num of degrees)  must = Dz_aug : "
                                                    f"{len(ETT_fits[d].tt.dims)} != {Dz_aug}")
        ETT_fits[d].tt.set_core(Dz_aug - 1)
        fd = ETT_fits[d](z_aug_domain_adjusted)
        fd_list.append(fd)
    f = torch.cat(tensors=fd_list, dim=1)
    return f


def generate_hybrid_cnf_trajectory(z_start: torch.Tensor, logp_zstart: torch.Tensor, h: float,
                                   ETT_fits: List[Extended_TensorTrain],
                                   nn_cnf_model: CNF, t_start: float, t_end: float, t_step: float,
                                   domain_stripe: List[float], adjoint: bool) -> dict:
    """

    :param z_start:
    :param logp_zstart:
    :param h:
    :param ETT_fits:
    :param nn_cnf_model:
    :param t_start:
    :param t_end:
    :param t_step:
    :param domain_stripe:
    :param adjoint:
    :return:
    """
    # assertions
    assert h > 0, "h must be > 0"
    assert abs(t_end - t_start) > h, "abs(t_end-t_start) must be > 0"
    # step 1 : make tt-rk-step
    # f = dz/dt
    f = ETT_fits_predict(ETT_fits=ETT_fits, z=z_start, t=t_start, domain_stripe=domain_stripe)
    trace_df_dz = ETT_fits_trace_grad(ETT_fits=ETT_fits, z=z_start, t=t_start, domain_stripe=domain_stripe)
    g = -trace_df_dz
    direction_num = 1 if t_end > t_start else -1
    assert direction_num * (t_end - t_start) > 0, "(t_end-t_start) and direction_num must be both negative or positive"
    z_h = z_start + direction_num * f * h
    logp_zh = logp_zstart.view(-1, 1) + direction_num * g.view(-1, 1) * h

    # step 2 : make cnf step from z_h to z_end
    # FIXME, is that the best way to do this ?
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    t_h = t_start + direction_num * h
    t_vals = torch.tensor(np.arange(t_h, t_end + direction_num * 0.01, t_step))
    zt, logp_zt = odeint(func=nn_cnf_model, y0=(z_h.type(torch.float32), logp_zh.type(torch.float32)), t=t_vals)
    # fixme, return the complete trajectory not the cnf part only
    return {'zt': zt, 'logp_zt': logp_zt, 't': t_vals}


def ETT_fits_trace_grad(ETT_fits: List[Extended_TensorTrain], z: torch.Tensor, t: float,
                        domain_stripe: List[float]) -> torch.Tensor:
    N = z.size()[0]
    Dz = z.size()[1]
    z_device = z.get_device() if z.is_cuda else torch.device('cpu')
    z_aug = torch.cat(tensors=[z, torch.tensor(t).repeat(N, 1).to(z_device)], dim=1)
    Dz_aug = Dz + 1

    z_aug_domain_adjusted = domain_adjust(x=z_aug, domain_stripe=domain_stripe)
    is_domain_adjusted(x=z_aug_domain_adjusted, domain_stripe=domain_stripe)
    # set of assertions
    assert is_domain_adjusted(x=z_aug_domain_adjusted, domain_stripe=domain_stripe)
    assert len(ETT_fits) == Dz
    df_dz_list = []
    for d in range(Dz):
        assert len(ETT_fits[d].tt.dims) == Dz_aug, (f"ETT order (num of degrees)  must = Dz_aug : "
                                                    f"{len(ETT_fits[d].tt.dims)} != {Dz_aug}")
        # TODO : is the set_core step necessary for grad call ?
        ETT_fits[d].tt.set_core(Dz_aug - 1)
        dfd_dz = ETT_fits[d].grad(x=z_aug_domain_adjusted)[:, :Dz]
        df_dz_list.append(dfd_dz)
    df_dz = torch.stack(df_dz_list, dim=1)
    trace_df_dz = torch.vmap(torch.trace)(df_dz)
    return trace_df_dz


def vanilla_cnf_optimize_step(optimizer: Optimizer, nn_cnf_model: torch.nn.Module, x: torch.Tensor, t0: float,
                              tN: float,
                              logp_diff_tN: torch.Tensor, adjoint: bool,
                              p_z0: torch.distributions.Distribution, device: torch.device) -> torch.Tensor:
    """

    :param device:
    :param optimizer:
    :param nn_cnf_model:
    :param x:
    :param t0:
    :param tN:
    :param logp_diff_tN:
    :param adjoint:
    :param p_z0:
    :return:
    """
    # 1) Zero the weights
    optimizer.zero_grad()
    # 2) Forward /odeint
    # FIXME, is that the best way to do this ?
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    z_t, logp_diff_t = odeint(
        nn_cnf_model,
        (x, logp_diff_tN),
        torch.tensor([tN, t0]).type(torch.float32).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )
    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    logp_x = p_z0.log_prob(z_t0).to(device) #- logp_diff_t0.view(-1)
    loss = -logp_x.mean(0)
    # 3) Backward (adjoint if odeint is attached to ode_adjoint)
    loss.backward()
    # 4) Update parameters
    optimizer.step()
    return loss


def hybrid_cnf_optimize_step(optimizer: Optimizer, nn_cnf_model: torch.nn.Module, x: torch.Tensor, t0: float,
                             tN: float, ETT_fits: List[Extended_TensorTrain],
                             logp_ztN: torch.Tensor, adjoint: bool,
                             p_z0: torch.distributions.Distribution, device: torch.device, h: float,
                             domain_stripe: List[float], itr: int, itr_threshold: int) -> torch.Tensor:
    """

    :param itr:
    :param domain_stripe:
    :param ETT_fits:
    :param logp_ztN:
    :param h:
    :param optimizer:
    :param nn_cnf_model:
    :param x:
    :param t0:
    :param tN:
    :param adjoint:
    :param p_z0:
    :param device:
    :return:
    """

    # FIXME, is that the best way to do this ?
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    # --- Start of function code ---
    if itr < itr_threshold:
        loss = vanilla_cnf_optimize_step(optimizer=optimizer, nn_cnf_model=nn_cnf_model, x=x, t0=t0, tN=tN,
                                         logp_diff_tN=logp_ztN, adjoint=adjoint, p_z0=p_z0, device=device)
        return loss
    if itr == itr_threshold:
        with torch.no_grad():
            # init ETTs
            ztN = x
            zt_, _ = odeint(func=nn_cnf_model, y0=(x, logp_ztN), t=torch.tensor([tN, tN - h]))
            ztN_minus_h = zt_[-1]
            yy = (ztN - ztN_minus_h) / h
            xx = ztN
            # xx_adj = domain_adjust(x=xx, domain_stripe=domain_stripe)
            # norm_before = get_avg_norm_ETTs(ETTs=ETT_fits)
            run_tt_als(x=xx, t=tN, y=yy, ETT_fits=ETT_fits, test_ratio=0.2, tol=1e-6, domain_stripe=domain_stripe)
            # norm_after = get_avg_norm_ETTs(ETTs=ETT_fits)
        # print('---')
    # 1) Init. opt the weights

    optimizer.zero_grad()
    # 2) Forward /odeint

    # the extra tt-rk-step
    ztN = x
    assert not ztN.requires_grad, "z(tN) require grad must be False"
    tt_rk_step_result = tt_rk_step(t=tN, h=h, z=ztN, log_p_z=logp_ztN, ETT_fits=ETT_fits, direction='backward',
                                   domain_stripe=domain_stripe)
    ztN_minus_h = tt_rk_step_result['z_prime']

    logp_ztN_minus_h = tt_rk_step_result['log_p_z_prime']
    # FIXME, debug code. To Remove
    rmse_z = MSELoss()(ztN, ztN_minus_h)
    rmse_logpz = MSELoss()(logp_ztN, logp_ztN_minus_h)
    # ---
    tN_minus_h = tN - h
    assert not ztN_minus_h.requires_grad, "the starting z(t_N - h) should have requires_grad = False"
    z_t, logp_diff_t = odeint(
        nn_cnf_model,
        (ztN_minus_h, logp_ztN_minus_h),
        torch.tensor([tN_minus_h, t0]).type(torch.float32).to(device),
        atol=1e-5,
        rtol=1e-5,
        method='dopri5',
    )
    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    log_p_z0_raw = p_z0.log_prob(z_t0)
    logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
    loss = -logp_x.mean(0)
    # 3) Backward (adjoint if odeint is attached to ode_adjoint)
    loss.backward()
    # 4) Update parameters
    optimizer.step()
    with torch.no_grad():
        # optimize TT
        zt_prime, _ = odeint(
            nn_cnf_model,
            (z_t0, log_p_z0_raw.to(device)),
            torch.tensor([t0, tN_minus_h]).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        z_tN_minus_h_prime = zt_prime[-1]
        rmse_ = MSELoss()(ztN_minus_h, z_tN_minus_h_prime)  # FIXME , for debug , to remove
        yy = (ztN - z_tN_minus_h_prime) / h
        xx = ztN
        # assert is_domain_adjusted(x=xx, domain_stripe=domain_stripe)
        tol = 1e-6
        norm_ETTs_before = get_avg_norm_ETTs(ETT_fits)
        tt_als_results = run_tt_als(x=xx, t=tN, y=yy, ETT_fits=ETT_fits, test_ratio=0.2,
                                    tol=tol, domain_stripe=domain_stripe)
        norm_ETTs_after = get_avg_norm_ETTs(ETT_fits)
        diff_ = norm_ETTs_before - norm_ETTs_after

    return loss


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


def validate_vanilla_cnf_trained_models(nn_cnf_model: torch.nn.Module,
                                        base_distribution: torch.distributions.Distribution,
                                        target_distribution: torch.distributions.Distribution,
                                        t0: float, tN: float,
                                        num_samples: int,
                                        adjoint: bool) -> dict:
    # FIXME, is that the best way to do this ?
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    # --- Start of function code ---
    z0_sample = base_distribution.sample(torch.Size([num_samples]))
    log_p_z0 = base_distribution.log_prob(z0_sample).view(-1, 1)

    zt, logpzt = odeint(func=nn_cnf_model, t=torch.tensor([t0, tN]), y0=(z0_sample, log_p_z0))
    ztN = zt[-1]
    normality_test = pg.multivariate_normality(X=ztN.detach().cpu().numpy())
    sample_mean = torch.mean(input=ztN, dim=0)
    sample_cov = torch.cov(input=ztN.T)
    mean_rmse = MSELoss()(sample_mean, target_distribution.mean).item()
    cov_rmse = MSELoss()(torch.diagonal(sample_cov), target_distribution.variance).item()

    if normality_test.normal:
        hz_loss = (1 - normality_test.pval) * (1 - mean_rmse) * (1 - cov_rmse)
    else:
        hz_loss = np.Inf
    assert isinstance(hz_loss, float)
    validation_results = {'hz_loss': hz_loss}
    # wd distance
    dist_mean = target_distribution.mean
    dist_cov = torch.diag(target_distribution.variance)
    wd = wasserstein_distance_two_gaussians(m1=dist_mean, m2=sample_mean, C1=dist_cov,
                                            C2=torch.round(sample_cov, decimals=1))
    mean_rmse = MSELoss()(dist_mean, sample_mean).item()
    cov_rmse = MSELoss()(dist_cov, sample_cov).item()
    validation_results['wd'] = wd
    validation_results['mean_rmse'] = mean_rmse
    validation_results['cov_rmse'] = cov_rmse
    return validation_results


def validate_hybrid_trained_models(nn_cnf_model: torch.nn.Module, ETT_fits: List[Extended_TensorTrain],
                                   base_distribution: torch.distributions.Distribution,
                                   target_distribution: torch.distributions.Distribution, t0: float, tN: float,
                                   num_samples: int, adjoint: bool, h: float, domain_stripe: List[float], itr: int,
                                   itr_threshold: int) -> dict:
    """

    :param nn_cnf_model:
    :param ETT_fits:
    :param base_distribution:
    :param target_distribution:
    :param t0:
    :param tN:
    :param num_samples:
    :param adjoint:
    :param h:
    :param domain_stripe:
    :param itr:
    :param itr_threshold:
    :return:
    """
    # FIXME, is that the best way to do this ?
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # --- Start of function code ---
    z0_sample = base_distribution.sample(torch.Size([num_samples]))
    log_p_z0 = base_distribution.log_prob(z0_sample).view(-1, 1)

    if itr < itr_threshold:
        zt, _ = odeint(func=nn_cnf_model, y0=(z0_sample, log_p_z0), t=torch.tensor([t0, tN]), )
        ztN = zt[-1]
    else:
        zt, logpzt = odeint(func=nn_cnf_model, t=torch.tensor([t0, tN - h]), y0=(z0_sample, log_p_z0))
        ztN_minus_h = zt[-1]
        logpztN_minus_h = logpzt[-1]
        # FIXME, modify the h and domain stripe params to be coming from outer scope calls

        results = tt_rk_step(t=t0, z=ztN_minus_h, log_p_z=logpztN_minus_h, h=h, ETT_fits=ETT_fits, direction="forward",
                             domain_stripe=domain_stripe)
        ztN = results['z_prime']
        # logpztN_minus_h = results['log_p_z_prime']

    normality_test = pg.multivariate_normality(X=ztN.detach().cpu().numpy())
    m = torch.mean(input=ztN, dim=0)
    cov_ = torch.cov(input=ztN.T)
    mean_rmse = MSELoss()(m, target_distribution.mean).item()
    cov_rmse = MSELoss()(torch.diagonal(cov_), target_distribution.variance).item()
    if normality_test.normal:
        hz_loss = (1 - normality_test.pval) * (1 - mean_rmse) * (1 - cov_rmse)
    else:
        hz_loss = np.Inf
    assert isinstance(hz_loss, float)
    validation_results = {'hz_loss': hz_loss}
    wd = wasserstein_distance_two_gaussians(m1=target_distribution.mean, m2=m,
                                            C1=torch.diag(target_distribution.variance), C2=cov_)
    validation_results['wd'] = wd
    validation_results['mean_rmse'] = mean_rmse
    validation_results['cov_rmse'] = cov_rmse
    return validation_results


def get_avg_norm_ETTs(ETTs: List[Extended_TensorTrain], aggregation: str = "sum") -> float:
    norms = []
    for i, ETT in enumerate(ETTs):
        norm = TensorTrain.frob_norm(ETT.tt.comps).item()
        norms.append(norm)
    if aggregation == "sum":
        return sum(norms)
    elif aggregation == "avg":
        return np.mean(norms)
    else:
        raise ValueError(f"Unknown aggregation : {aggregation}")


"""
Some experimental notes for the function lnode.utils.hybrid_cnf_optimize_step 

Training the NN trajectory while Keeping the TT part fixed , with different h values for the TT-RK step

h       niter       loss(begin)         loss(end)
0       1000        4.07                0.34
1       1000        17.12               2.14
2       1000        47.7                3.613

---
For vanilla CNF :   lnode.utils.vanilla_cnf_optimize_step
h       niter       loss(begin)         loss(end)
*       1000        4.07                0.34

Note that for h=0 in hybrid(lnode) with h=0 and vanilla, the loss(begin) and loss(end) are equal
"""
