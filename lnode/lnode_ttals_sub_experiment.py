"""
This script is for running Sub-Experiment to run a TT-ALS training:

\begin{equation*}
    \bm{\mathscr{W}_{TT}} = argmin_{\bm{\mathscr{W}}_{TT}}
    ||\mathbf{z}_{d}(t_0+h)-\bm{\mathscr{W}}^{d}_{TT}\bm{\tilde{\Phi}}^{poly}_{\mathcal{R}=1}([\mathbf{z}(t_0),t_0])||_2^2
\end{equation*}

The generated sub-trajectory [ z(t_0) -> z(t_0 + h ) ] is based on a generated vanilla-CNF model

The purpose of this sub-experiment is to test the train-ability of sub-trajectory based on
TT-ALS training method

Example CMD line argument for running
============================================
export PYTHONPATH="${PYTHONPATH}:./torchdiffeq/GMSOC"
python3 lnode/lnode_ttals_sub_experiment.py --artifact "lnode/artifacts/vanilla_2023-07-20T13:36:11.559464_dist_MultivariateNormal_d_4_niters_1000.pkl" --trajectory-opt "vanilla" --device "cpu" --rank 2 --degree 3 --h-steps 3 --tol 1e-4
"""
import argparse
import logging
import pickle
import random
from typing import List

import pingouin as pg
import numpy as np
import torch
from torch.nn import MSELoss

from utils import is_domain_adjusted, tt_rk_step, generate_hybrid_cnf_trajectory
from GMSOC.functional_tt_fabrique import orthpoly, Extended_TensorTrain
from examples.models import CNF
from utils import domain_adjust

# SEED
SEED = 38473923  # FIXME , see other valid seed values for reproducing results
np.random.seed(SEED)
random.seed(SEED)
torch.random.manual_seed(SEED)

# Global Variables
EPS = 1e-4
# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--artifact', type=str, required=True)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--trajectory-opt', type=str, choices=['vanilla', 'hybrid'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], required=True)
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--degree', type=int, required=True)
parser.add_argument('--h-steps', type=int, required=True)
parser.add_argument('--tol', type=float, required=True)
parser.add_argument('--n-samples', type=int, required=True)
# pass a list as cmd args
# https://stackoverflow.com/a/32763023/5937273
parser.add_argument('--domain-stripe', nargs='*', type=float, default=[-1, 1], help='domain to adjust target R.V to')

# parser.add_argument('--t0-index', type=int, required=True)
args = parser.parse_args()

assert isinstance(args.domain_stripe, list) and len(args.domain_stripe) == 2 and args.domain_stripe[0] < \
       args.domain_stripe[1], \
    f"domain must be of type list, len(list) = 2 and domain[0] < domain[1] , however we got domain = {args.domain}"

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device
if args.device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.torch.device('cuda:' + str(args.gpu)
                                if torch.cuda.is_available() else 'cpu')
logger.info(f'Is CUDA device available? = {torch.cuda.is_available()}')
logger.info(f'Device = {device}')


# noinspection PyShadowingNames


# FIXME to delete this piece of code
# def generate_hybrid_zt_trajectory(z_tN: torch.Tensor, tN: float, t0: float, t_step: float, h: float,
#                                   cnf_trajectory_model: CNF, ETT_fits: List[Extended_TensorTrain],
#                                   domain_stripe: List[float]) -> dict:
#     # sub-step 1 : generate first segment for the trajectory z(tN-h) from z(tN) using fitted ETT_fits
#
#     results_ = tt_rk_step(tN=tN, z_tN=z_tN, h=h, ETT_fits=ETT_fits, domain_stripe=domain_stripe)
#     z_tN_minus_h_pred_ = results['z_tN_minus_h_pred']
#     # TODO
#     """
#     How to calculate logp at t= tN-h using the tt-rk-step ?? , then compare it with the vanilla-CNF one
#     """
#     # subs-step 2 : generate second segment of the trajectory
#     tN_minus_h_ = tN - h
#     t_vals_ = torch.tensor(np.arange(tN_minus_h_, t0 - t_step, -t_step))
#
#     z_t_trajectory, logp_diff_trajectory = odeint(func=cnf_trajectory_model, y0=())


def run_tt_als(x: torch.Tensor, t: float, y: torch.Tensor, poly_degree: int, rank: int, test_ratio: float,
               tol: float, domain_stripe: List[float]) -> dict:
    """

    :param domain_stripe:
    :param x: pure x, no time
    :param t:
    :param y:
    :param poly_degree:
    :param rank:
    :param test_ratio:
    :param tol:
    :return:
    """
    N = x.shape[0]
    N_test = int(test_ratio * N)
    N_train = N - N_test

    x_aug = torch.cat(tensors=[x, torch.tensor([t]).repeat(N, 1)], dim=1)
    Dx_aug = x_aug.shape[1]
    Dy = y.shape[1]
    #
    degrees = [poly_degree] * Dx_aug
    ranks = [1] + [rank] * (Dx_aug - 1) + [1]
    # order = len(degrees)  # not used, but for debugging only
    domain = [domain_stripe for _ in range(Dx_aug)]
    op = orthpoly(degrees, domain)
    x_aug_domain_adjusted = domain_adjust(x=x_aug, domain_stripe=domain_stripe)
    is_domain_adjusted(x=x_aug_domain_adjusted, domain_stripe=domain_stripe)
    ETT_fits = [Extended_TensorTrain(op, ranks) for _ in range(Dy)]
    y_predicted_list = []
    mse_loss_fn = torch.nn.MSELoss()
    for j in range(Dy):
        y_d = y[:, j].view(-1, 1)
        # ALS parameters
        reg_coeff = 1e-2
        iterations = 40
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

        train_rmse = torch.sqrt(mse_loss_fn(y_d_predict_train, y_d.type(torch.float64)[:N_train, :]))
        test_rmse = torch.sqrt(mse_loss_fn(y_d_predict_val, y_d.type(torch.float64)[N_train:, :]))
        logger.info(f'For j ( d-th dim of y)  = {j} :TT-ALS RMSE on training set = {train_rmse}')
        logger.info(f'For j (d-th dim of y)  = {j}: TT-ALS RMSE on test set = {test_rmse}')
        #
        y_d_predicted = ETT_fits[j](x_aug_domain_adjusted.type(torch.float64))
        y_predicted_list.append(y_d_predicted)
        logger.info("======== Finished TT-ALS training ============")
    y_predicted = torch.cat(tensors=y_predicted_list, dim=1)
    prediction_tot_rmse = torch.sqrt(mse_loss_fn(y, y_predicted))
    return {'ETT_fits': ETT_fits, 'prediction_tot_rmse': prediction_tot_rmse}


# FIXME ToDelete The following function
# def verify_fit_ETT(ETTs_fit: List[Extended_TensorTrain], cnf_trajectory_model: CNF,
#                    target_distribution: torch.distributions.Distribution,
#                    base_distribution: torch.distributions.Distribution, t0: float, tN: float, n_samples: int,
#                    h: float, tN_minus_h: float) -> float:
#     """
#     In words, this function verify that  : Given a fitted ETT model (TT + Basis) and a fitted CNF trajectory model
#     ( cnf_trajectory model) a sample x following a base distribution p_X can be
#     normalized from a sample Y following distribution p_Y by solvig ODE over trajectory tN->t0:
#
#         i) x = z(t0) , y = z(tN)
#         ii) from z(tN) to z(tN-h) : one explicit RK step ( euler step)
#             z(tN) = y ~ target_distribution , sampling
#             z(tN-h) = z(tN) + \int_{tN}^{tN-h} (f_TT(z(tN,tN)) dt
#             z(tN-h) = z(tN) + (-h) f_TT(z(tN),tN)
#         iii) from z(tN-h) to z(t0)
#             z(t0) = odeint(z(tN-h) , tN-h -> t0, cnf_trajectory_model)
#         iv) calculate log_prob for the base_distribution give z(t0) , should be close to zero
#     :return: log_prob_base_dist_z0  log_prob(base_distribution | z0)
#     """
#     # step i)
#
#     y = target_distribution.sample(torch.Size([n_samples]))  # [z_tN]
#     logp_diff_tN = torch.zeros(n_samples, 1).type(torch.float32).to(device)
#     # fixme : start sanity check (0)
#     y_np = y.detach().numpy()
#     normality_test_results_0 = pg.multivariate_normality(X=y_np)
#     # fixme : end sanity check (0)
#     # ---
#     # fixme : start sanity check (1)
#     zt, _ = odeint(func=trajectory_model, y0=(y, logp_diff_tN),
#                    t=torch.tensor([tN_minus_h, t0]))
#     zt0_hat = zt[-1]
#     zt0_hat_np = zt0_hat.detach().numpy()
#     normality_test_results_1 = pg.multivariate_normality(X=zt0_hat_np)
#     # fixme : end sanity check (1)
#
#     Dy = y.shape[1]
#     y_aug = torch.cat([y, torch.tensor([tN]).repeat(n_samples, 1)], dim=1)  # [z_tN,tN]
#     y_aug_domain_adjusted = domain_adjust(x=y_aug, domain_stripe=[-1, 1])  # fixme, make parametric
#     y_domain_adjusted_min = torch.min(y_aug_domain_adjusted, dim=0)
#     y_domain_adjusted_max = torch.max(y_aug_domain_adjusted, dim=0)
#     dzdt_list = []
#     for j in range(Dy):
#         ETTs_fit[j].tt.set_core(Dy - 1)
#         dzdt_j = ETTs_fit[j](y_aug_domain_adjusted)  # applied to [z_tN,tN]
#         dzdt_list.append(dzdt_j)
#     dzdt_tensor = torch.cat(dzdt_list, dim=1)
#     z_tN_domain_adjusted = y_aug_domain_adjusted[:, :Dy]
#     z_tN_domain_adjusted_min = torch.min(z_tN_domain_adjusted, dim=0)
#     z_tN_domain_adjusted_max = torch.max(z_tN_domain_adjusted, dim=0)
#     z_tN_minus_h = z_tN_domain_adjusted - h * dzdt_tensor
#     # fixme : start sanity check (2)
#     z_tN_minus_h_np = z_tN_minus_h.detach().numpy()
#     normality_test_results_2 = pg.multivariate_normality(X=z_tN_minus_h_np)
#     # fixme : end sanity check (2)
#     # step iii)
#     assert abs(h - (tN - float(tN_minus_h))) < 0.001
#
#     # z_tN_minus_h.type(torch.float32)
#     z_t_hybrid, _ = odeint(func=trajectory_model, y0=(z_tN_minus_h.type(torch.float32), logp_diff_tN),
#                            t=torch.tensor([tN_minus_h, t0]))
#     z_t0_hat_hybrid = z_t_hybrid[-1]
#     # fixme : start sanity check(3)
#     z_t0_hybrid_hat_np = z_t0_hat_hybrid.detach().numpy()
#     normality_test_results_3 = pg.multivariate_normality(X=z_t0_hybrid_hat_np)
#     # fixme : end sanity check(3)
#     # TODO : might be that domain adjustment distort the distribution\
#     # ---
#     # fixme : start sanity check (4)
#     z_t_domain_adjusted, _ = odeint(func=trajectory_model, y0=(z_tN_domain_adjusted.type(torch.float32), logp_diff_tN),
#                                     t=torch.tensor([tN, t0]))
#     z_tN_no_adjustment = y[:, :Dy]
#     z_t_no_adjustment, _ = odeint(func=trajectory_model, y0=(z_tN_no_adjustment.type(torch.float32), logp_diff_tN),
#                                   t=torch.tensor([tN, t0]))
#     normality_test_results_4_1 = pg.multivariate_normality(X=z_t_domain_adjusted[-1].detach().numpy())
#     normality_test_results_4_2 = pg.multivariate_normality(X=z_t_no_adjustment[-1].detach().numpy())
#     # fixme : end sanity check (4)
#     """
#     Status
#     It seems that domain adjustment distort the distribution when going through the CNF Trajectory, see
#     normality_test 4_1 and 4_2
#     https://drive.google.com/file/d/1WV6HQdByzlxauRGUTEdqgZ2w-m-RMK0H/view?usp=drive_link
#
#     # One possible solution : train Trajectory model on domain adjusted data ?
#     steps : (for ordinary CNF)
#     i) Generate Samples from target distribution
#     ii) Make domain adjustment ( -1 to 1)
#     iii) Retest normality for domain adjustment
#     iv) Train CNF
#     v) Train TT-ALS over h segment
#     vi) Re-test the hybrid trajectory generation
#
#     """
#     logger.info(f'Verification Finished')


if __name__ == '__main__':
    artifact = pickle.load(
        open(f'{args.artifact}', "rb"))
    dim = artifact['dim']
    hidden_dim = artifact['args']['hidden_dim']
    width = artifact['args']['width']
    trajectory_model = CNF(in_out_dim=dim, hidden_dim=hidden_dim, width=width, device=device).type(torch.float32)
    trajectory_model.load_state_dict(artifact['model'])
    logger.info(f'Successfully loaded CNF model and Meta data')

    # Verify normality of generated data
    ## step(1). Generate samples out of the loaded model and meta-data
    ##  then generate z(t0) by odeint z(tN) -> z(t0) and test z(t0) for normality
    z_tN = artifact['target_distribution'].sample(torch.Size([args.n_samples])).to(device)
    logp_ztN_dt = torch.zeros(args.n_samples, 1).type(torch.float32).to(device)
    t0 = artifact['args']['t0']
    tN = artifact['args']['t1']
    t_vals_1 = torch.tensor(list(np.arange(tN, t0 - 1, -1)))
    logger.info(f'Running CNF trajectory')
    z_t_trajectory_1, logp_dzdt_trajectory_1 = odeint(func=trajectory_model, y0=(z_tN, logp_ztN_dt), t=t_vals_1)
    z_t0_1 = z_t_trajectory_1[-1]
    logp_zt0_pred_1 = logp_dzdt_trajectory_1[-1]
    ## Test the z(t0) generated from step(1) for normality and matching mio and Sigma
    z_t0_np_1 = z_t0_1.detach().numpy()
    normality_test = pg.multivariate_normality(X=z_t0_np_1, alpha=0.1)
    assert normality_test.normal, ("z(t0) Generated using vanilla-CNF from z(t_N) ~ "
                                   "N(mio,Sigma) -> z(t_0) ^N(0,a.I) is not Normal")
    logger.info('Finished testing vanilla CNF Generative path z(tN) ~ N(mio,Sigma) -> z(t0) ~ N(0,a.I) : steps 1 and 2')
    ## step(2) get z(tN-h) and generate z(t0) using vanilla-CNF from z(tN-h) -> z(t0)
    tN_minus_h = t_vals_1[args.h_steps].item()
    z_tN_minus_h = z_t_trajectory_1[args.h_steps]
    logp_diff_tN_minus_h = logp_dzdt_trajectory_1[args.h_steps]
    h = tN - tN_minus_h
    assert h > 0, "h must be > 0"
    t_vals_2 = torch.tensor(list(np.arange(tN_minus_h, t0 - 1, -1)))
    z_t_trajectory_2, logp_diff_2 = odeint(func=trajectory_model, y0=(z_tN_minus_h, logp_diff_tN_minus_h), t=t_vals_2)
    z_t0_2 = z_t_trajectory_2[-1]
    diff_ = torch.norm(z_t0_1 - z_t0_2)
    logger.info(f'Finished testing vanilla CNF Generative path  z(tN-h) ~ N(mio,Sigma) -> z(t0) ~ N(0,a.I) : step 2')

    ## step(3)
    # TT-ALS to find a model that maps z(tN) to z(tN-h) via RK-step (euler step)
    yy = (z_tN_minus_h - z_tN) / (-h)
    xx = z_tN
    tt_rk_backward_step_results = run_tt_als(x=xx, t=tN, y=yy, poly_degree=args.degree, rank=args.rank, test_ratio=0.2,
                                             tol=args.tol,
                                             domain_stripe=args.domain_stripe)
    ETT_fits = tt_rk_backward_step_results['ETT_fits']
    prediction_tot_rmse = tt_rk_backward_step_results['prediction_tot_rmse']
    logger.info(
        f'Finished TT-ALS with y=(z_tN_minus_h - z_tN) / (-h) and x = z_tN with prediction_tot_rmse  = '
        f'{prediction_tot_rmse}')
    ## Step(4)
    # make a backward tt-rk step
    tt_rk_backward_step_results = tt_rk_step(t=tN, z=z_tN, log_p_z=logp_ztN_dt, ETT_fits=ETT_fits, h=h,
                                             domain_stripe=args.domain_stripe, direction="backward")
    f = tt_rk_backward_step_results['dzdt']
    z_tN_minus_h_pred = tt_rk_backward_step_results['z_prime']

    mse_val = MSELoss()(z_tN_minus_h_pred, z_tN_minus_h)
    assert mse_val <= EPS, (f"z(tN-h) calculated from vanilla-CNF and tt-rk-step are not "
                            f"close-enough. mse_error = {mse_val}")
    logger.info(
        f'Successfully Finished step(4) : '
        f'Generated Close-Enough z(tN-h) from vanilla-CNF and tt-rk-euler step from z(tN) . MSE = {mse_val}')

    ## Step(5)
    # Similar to step (2), except that z(tN-h) is not extracted from vanilla-CNF trajctory but from
    # TT-ALS fitted model prediction
    logger.info('Running Step 5 : Generate Hybrid Trajectory from z(tN) to z(t0) using TT-RK-Euler '
                'step from z(tN) to z(tN-h) and vanilla CNF from z(tN-h) to z(t0) ')
    log_p_ztN_minus_h_pred = tt_rk_backward_step_results['log_p_z_prime']
    z_t_trajectory_hybrid, logp_diff_hybrid = odeint(func=trajectory_model,
                                                     y0=(z_tN_minus_h_pred.type(torch.float32),
                                                         log_p_ztN_minus_h_pred.type(torch.float32)), t=t_vals_2)
    z_t0_hybrid = z_t_trajectory_hybrid[-1]
    normality_test_z_t0_hybrid = pg.multivariate_normality(X=z_t0_hybrid.detach().cpu().numpy())
    mse_val = MSELoss()(z_t0_hybrid, z_t0_1)
    assert mse_val < EPS, f"z(t0)_hybrid is not close enough to z(t0)_vanilla , MSE = {mse_val}"
    logger.info('Successfully finished step (5): flat code to generate z(t0) from hybrid trajectory (TT-RK + NN-CNF) '
                'from (ztN) thru z(tN-h) and should be close enough to z(t0) generated from z(tN) via vanilla-CNF')
    #
    logger.info(f'Starting Step(6) : Testing Hybrid Trajectory Generation function from tN = {tN} to t0 = {t0}')
    results = generate_hybrid_cnf_trajectory(z_start=z_tN, logp_zstart=logp_ztN_dt, h=h,
                                             ETT_fits=ETT_fits,
                                             nn_cnf_model=trajectory_model,
                                             t_start=tN,
                                             t_end=t0,
                                             t_step=-1,
                                             domain_stripe=args.domain_stripe,
                                             adjoint=args.adjoint)
    # {'zt': zt, 'logp_zt': logp_zt, 't': t_vals}
    zt_hybrid_trajectory = results['zt']
    logp_zt = results['logp_zt']
    t_hybrid = results['t']

    z_t0_hybrid_2 = zt_hybrid_trajectory[-1]
    log_zt0_hybrid = logp_zt[-1]
    mse_val_z0_hybrid = MSELoss()(z_t0_hybrid_2, z_t0_1)
    mse_val_logp_z0_hybrid = MSELoss()(log_zt0_hybrid, logp_zt0_pred_1)
    assert mse_val_z0_hybrid < EPS  # and mse_val_logp_z0_hybrid < EPS
    # TODO should we check MSE for log also, does it make sense ?
    logger.info(
        f'Finished Step(6) with mse(z(t0)) = {mse_val_z0_hybrid} and mse(log(p(z0))) = {mse_val_logp_z0_hybrid}')

    ## Generate the
    #
    # # step(2). verify the distribution of the generated data
    # ## 2.1 Using HZ test
    # z_t0_hat = z_t[-1]
    # z_t0_np = z_t0_hat.detach().cpu().numpy()
    # normality_test_results = pg.multivariate_normality(X=z_t0_np)
    # logger.info(f'Normality test results = {normality_test_results}')
    # sample_mio = z_t0_hat.mean(0)
    # sample_sigma = torch.cov(z_t0_hat.T)
    # ## 2.2 verify the parameters of the generated data
    # mean_abs_err = torch.norm(sample_mio - artifact['base_distribution'].mean.detach().cpu())
    # mean_rel_err = mean_abs_err / torch.norm(
    #     artifact['target_distribution'].mean)
    # cov_abs_err = torch.norm(sample_sigma - artifact['base_distribution'].covariance_matrix.detach().cpu())
    # cov_rel_err = cov_abs_err / torch.norm(
    #     artifact['target_distribution'].covariance_matrix)
    # logger.info(f'mean_abs_err = {mean_abs_err}')
    # logger.info(f'mean_rel_err = {mean_rel_err}')
    # logger.info(f'cov_abs_err = {cov_abs_err}')
    # logger.info(f'cov_rel_err = {cov_rel_err}')
    # ## 2.3 calculate log-prob
    # dist_device = artifact['base_distribution'].mean.device
    # log_prob = artifact['base_distribution'].log_prob(z_t0_hat.to(dist_device)).mean(0)
    # prob_ = torch.exp(log_prob)
    # # benchmark with a sample
    # z_t0_benchmark_sample = artifact['base_distribution'].sample(torch.Size([z_t0_hat.shape[1]]))
    # log_prob_benchmark = artifact['base_distribution'].log_prob(z_t0_benchmark_sample).mean(0)
    #
    # # 3. Run TT-ALS for the RK-step : z(tN) -> z(tN-h)
    # t_N = t_vals[0]
    # tN_minus_h = t_vals[args.h_steps]
    # h = t_N - tN_minus_h
    # # mind the signs !!
    # z_tN = z_t[0]
    # z_tN_minus_h = z_t[args.h_steps]

    # ETTs_fit = run_tt_als(x=x, y=y, t=t_N, poly_degree=args.degree, rank=args.rank, test_ratio=0.2, tol=args.tol)
    # logger.info('Fitting TT-ALS for the sub-trajectory finished')
    # verify_fit_ETT(ETTs_fit=ETTs_fit, cnf_trajectory_model=trajectory_model,
    #                target_distribution=artifact['target_distribution'], base_distribution=artifact['base_distribution'],
    #                t0=artifact['args']['t0'], tN=artifact['args']['t1'], n_samples=args.n_samples, h=h,
    #                tN_minus_h=tN_minus_h)
    # #
    # # # double test with hybrid trajectory
    # # ## 1. Explicit Euler step
    # # # z(tN-h) = ode_int(z(tN),[tN -> tN-h], f(z(tN),tN))
    # # # apply explicit euler one-step
    # # # z(tN-h) = z(tN) + \int_{tN}^{tN-h} f(z(tN),tN))
    # # # fixme, make domain stripe parametric
    # # N = x.shape[0]
    # # x = z_tN
    # # x = torch.cat([x, torch.tensor([tN]).repeat(N, 1)], dim=1)
    # # x_domain_adjusted = adjust_tensor_to_domain(x=x, domain_stripe=[-1, 1])
    # # x_domain_adjusted_min = torch.min(x_domain_adjusted, dim=0)
    # # x_domain_adjusted_max = torch.max(x_domain_adjusted, dim=0)
    # # Dx = x.shape[1]
    # # ETT_fit.tt.set_core(Dx - 1)
    # # f = ETT_fit(x=x_domain_adjusted)
    # # z_tN_minus_h = z_tN - h * f  # euler step, but backward
    # # t_vals = torch.tensor([t_N_minus_h, t0])
    # # z_t, _ = odeint(func=trajectory_model, y0=(z_tN_minus_h.type(torch.float32), logp_diff_tN), t=t_vals)
    # # z_t0_hat = z_t[-1]
    # # logger.info(f'Finished hybrid trajectory computation')
