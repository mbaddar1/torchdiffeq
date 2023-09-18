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
import sys
from typing import List

import pingouin as pg
import numpy as np
import torch
from torch.nn import MSELoss

from utils import is_domain_adjusted, tt_rk_step, generate_hybrid_cnf_trajectory, run_tt_als, get_ETTs
from GMSOC.functional_tt_fabrique import orthpoly, Extended_TensorTrain
from examples.models import CNF
from utils import domain_adjust

# SEED
SEED = 38473923  # FIXME , see other valid seed values for reproducing results
np.random.seed(SEED)
random.seed(SEED)
torch.random.manual_seed(SEED)

# Global Variables
EPS = 1e-3
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

    # Verify the normality of generated data
    ## step(1). Generate samples out of the loaded model and meta-data
    ##  then generate z(t0) by odeint z(tN) -> z(t0) and test z(t0) for normality
    z_tN = artifact['target_distribution'].sample(torch.Size([args.n_samples])).to(device)
    logp_ztN_dt = torch.zeros(args.n_samples, 1).type(torch.float32).to(device)
    t0 = artifact['args']['t0']
    tN = artifact['args']['t1']
    t_vals_vanilla = torch.tensor(list(np.arange(tN, t0 - 1, -1)))
    logger.info(f'Step 1 : Running vanilla CNF/ODE trajectory generation from z(tN) to z(t0) ')
    z_t_trajectory_vanilla, logp_dzdt_trajectory_vanilla = (
        odeint(func=trajectory_model, y0=(z_tN, logp_ztN_dt), t=t_vals_vanilla))
    z_t0_vanilla = z_t_trajectory_vanilla[-1]
    logp_zt0_pred_vanilla = logp_dzdt_trajectory_vanilla[-1]
    ## Test the z(t0) generated from step(1) for normality and matching mio and Sigma
    z_t0_np_vanilla = z_t0_vanilla.detach().numpy()
    normality_test = pg.multivariate_normality(X=z_t0_np_vanilla, alpha=0.1)
    assert normality_test.normal, ("z(t0) Generated using vanilla-CNF from z(t_N) ~ "
                                   "N(mio,Sigma) -> z(t_0) ^N(0,a.I) is not Normal")

    mean_log_p_z0_vanilla = torch.mean(artifact['base_distribution'].
                                       log_prob(z_t0_vanilla.to(torch.device('cuda:0'))), dim=0).item()
    logger.info(f'Step 1: mean log-prob for z0 generated from vanilla CNF = {mean_log_p_z0_vanilla}')
    logger.info('Step 1: Finished')
    ## ******************

    # FIXME, to remove Step(*) redundant to remove
    #  get z(tN-h) and generate z(t0) using vanilla-CNF from z(tN-h) -> z(t0)
    # logger.info('Step 2 : Do ODESOLVE from Z9')
    # tN_minus_h = t_vals_1[args.h_steps].item()
    # z_tN_minus_h = z_t_trajectory_vanilla[args.h_steps]
    # logp_diff_tN_minus_h = logp_dzdt_trajectory_vanilla[args.h_steps]
    # h = tN - tN_minus_h
    # assert h > 0, "h must be > 0"
    # # naming after step-2
    # t_vals_2 = torch.tensor(list(np.arange(tN_minus_h, t0 - 1, -1)))
    # z_t_trajectory_2, logp_diff_2 = odeint(func=trajectory_model, y0=(z_tN_minus_h, logp_diff_tN_minus_h),
    #                                        t=t_vals_2)
    # z_t0_2 = z_t_trajectory_2[-1]
    # rmse_2 = torch.sqrt(MSELoss()(z_t0_vanilla,z_t0_2)).item()
    # logger.info(f'Step')

    ## step(2)
    # TT-ALS to find a model that maps z(tN) to z(tN-h) via RK-step (euler step)
    logger.info('Step 2 : TT-ALS between y_tt = (z_tN-z_tN_minus_h) / (h) and x_tt = [z_tN,t_N]')
    logp_diff_tN_minus_h = logp_dzdt_trajectory_vanilla[args.h_steps]
    tN_minus_h = t_vals_vanilla[args.h_steps].item()
    z_tN_minus_h_vanilla = z_t_trajectory_vanilla[args.h_steps]
    h = tN - tN_minus_h
    assert h > 0, "h must be > 0"
    yy = (z_tN - z_tN_minus_h_vanilla) / h
    xx = z_tN
    D_in = xx.size()[1] + 1
    D_out = xx.size()[1]
    xx_device = xx.get_device() if xx.is_cuda else torch.device('cpu')
    ETT_fits = get_ETTs(D_in=D_in, D_out=D_out, rank=args.rank, domain_stripe=args.domain_stripe,
                        poly_degree=args.degree, device=xx_device)
    tt_als_results = run_tt_als(x=xx, t=tN, y=yy, ETT_fits=ETT_fits, test_ratio=0.2, tol=args.tol,
                                domain_stripe=args.domain_stripe)
    # ETT_fits = tt_als_results['ETT_fits']
    prediction_tot_rmse = tt_als_results['prediction_tot_rmse']
    logger.info(
        f'Finished TT-ALS with y=(z_tN-z_tN_minus_h) / (h) and x = z_tN with prediction_tot_rmse  = '
        f'{prediction_tot_rmse}\n'
        f'===============================================')
    ## ******************

    # Step(3)
    # make a backward tt-rk step
    logger.info('Step (3) : Backward TT-RK(euler) step')
    tt_rk_step_results = tt_rk_step(t=tN, z=z_tN, log_p_z=logp_ztN_dt, ETT_fits=ETT_fits, h=h,
                                    domain_stripe=args.domain_stripe, direction="backward")
    f = tt_rk_step_results['dzdt']
    z_tN_minus_h_pred = tt_rk_step_results['z_prime']

    rmse_val_3 = torch.sqrt(MSELoss()(z_tN_minus_h_pred, z_tN_minus_h_vanilla)).item()
    assert rmse_val_3 <= EPS, (f"z(tN-h) calculated from vanilla-CNF and tt-rk-step are not "
                               f"close-enough:  RMSE = {rmse_val_3} > EPS = {EPS}")
    logger.info(
        f'Successfully Finished step(3) : with RMSE(z(tN-h)_vanilla,'
        f'z(tN-h)_tt_rk = {rmse_val_3} <= EPS = {EPS}\n=======================================')

    ## Step(4)
    # Hybrid trajectory Generation
    # i) z(tN)->z(tN-h) using TT-RK(Euler) Step
    # # ii) z(tN-h) using vanilla CNF model
    # # note : t_{N-1} = t_N - h , i.e. h= t_N-t_{N-1}
    # logger.info('Running Step 4 : Generate Hybrid Trajectory from z(tN) to z(t0) : first using TT-RK-Euler '
    #             'step from z(tN) to z(tN-h) then vanilla CNF from z(tN-h) to z(t0) ')
    # t_vals_4 = torch.tensor(list(np.arange(tN_minus_h, t0 - 1, -1))) # named after the step number
    # log_p_ztN_minus_h_pred = tt_als_results['log_p_z_prime']
    # z_t_trajectory_hybrid, logp_diff_hybrid = odeint(func=trajectory_model,
    #                                                  y0=(z_tN_minus_h_pred.type(torch.float32),
    #                                                      log_p_ztN_minus_h_pred.type(torch.float32)),
    #                                                  t=t_vals_4)
    # z_t0_hybrid = z_t_trajectory_hybrid[-1]
    # normality_test_z_t0_hybrid = pg.multivariate_normality(X=z_t0_hybrid.detach().cpu().numpy())
    # rmse_val_4 = torch.sqrt(MSELoss()(z_t0_hybrid, z_t0_vanilla))
    # assert rmse_val_4 < EPS, (f"z(t0)_hybrid is not close enough to z(t0)_vanilla , "
    #                           f"RMSE = {rmse_val_4} > EPS = {EPS}")
    #
    # logger.info(f'Successfully finished step (4) with RMSE = {rmse_val_4} <= EPS = {EPS}')
    # #
    # logger.info(f'Starting Step(6) : Testing Hybrid Trajectory Generation function from tN = {tN} to t0 = {t0}')
    results = generate_hybrid_cnf_trajectory(z_start=z_tN, logp_zstart=logp_ztN_dt, h=h,
                                             ETT_fits=ETT_fits,
                                             nn_cnf_model=trajectory_model,
                                             t_start=tN,
                                             t_end=t0,
                                             t_step=-1,
                                             domain_stripe=args.domain_stripe,
                                             adjoint=args.adjoint)
    # {'zt': zt, 'logp_zt': logp_zt, 't': t_vals}
    logger.info('Running Step 4 : Generate Hybrid Trajectory from z(tN) to z(t0) : first using TT-RK-Euler '
                'step from z(tN) to z(tN-h) then vanilla CNF from z(tN-h) to z(t0) ')
    zt_hybrid_trajectory = results['zt']
    logp_zt = results['logp_zt']
    t_hybrid = results['t']
    z_t0_hybrid = zt_hybrid_trajectory[-1]
    log_zt0_hybrid = logp_zt[-1]
    rmse_val_z0_hybrid = torch.sqrt(MSELoss()(z_t0_hybrid, z_t0_vanilla))
    # mse_val_logp_z0_hybrid = MSELoss()(log_zt0_hybrid, logp_zt0_pred_vanilla)
    assert rmse_val_z0_hybrid < EPS, (f"RMSE for z(t0) generated from vanilla CNF and "
                                      f"Hybrid TT+NN CNF = {rmse_val_z0_hybrid} > EPS = {EPS}")

    logger.info(f"Finished Step (4) successfully : RMSE for z(t0) generated from vanilla CNF and "
                f"Hybrid TT+NN CNF = {rmse_val_z0_hybrid} > EPS = {EPS}")

    # Step (5) compare log-prob

    logger.info('Step (5) : Evaluating log(p(z(t0)) from vanilla and Hybrid CNF')

    mean_log_p_z0_hybrid = torch.mean(artifact['base_distribution']
                                      .log_prob(z_t0_hybrid.to(torch.device('cuda:0'))), dim=0).item()
    logger.info(f'log(p(z(t0))) vanilla and hybrid = {mean_log_p_z0_vanilla}, {mean_log_p_z0_hybrid}')
    logger.info(f'Step(5) Finished\n=============================')
    # Step 6 Evaluating the closeness between sample z(t0) vanilla , hybrid and the base distribution
    logger.info(
        f'Starting Step(6) :Evaluating the closeness between sample z(t0) '
        f'vanilla , hybrid and the base distribution')
    # vanilla
    normality_test_z0_vanilla = pg.multivariate_normality(z_t0_vanilla.detach().numpy())
    sample_mean_z0_vanilla = torch.mean(z_t0_vanilla, dim=0)
    sample_cov_z0_vanilla = torch.cov(z_t0_vanilla.T)

    logger.info(f'Normality test results for z(0) vanilla = {normality_test_z0_vanilla}')
    z0_vanilla_mean_rmse = torch.sqrt(
        MSELoss()(sample_mean_z0_vanilla,
                  artifact['base_distribution'].mean.to(torch.device('cpu'))))
    z0_vanilla_cov_rmse = torch.sqrt(
        MSELoss()(torch.diagonal(sample_cov_z0_vanilla),
                  artifact['base_distribution'].variance.to(torch.device('cpu'))))
    logger.info(f'RMSE between base-dist. mean and sample mean for z(t0) vanilla = {z0_vanilla_mean_rmse}')
    logger.info(f'RMSE between base-dist. variance and sample variance for z(t0) vanilla = {z0_vanilla_cov_rmse}')

    logger.info('---')
    # hybrid
    normality_test_z0_hybrid = pg.multivariate_normality(z_t0_hybrid.detach().numpy())
    logger.info(f'Normality test results for z(0) hybrid = {normality_test_z0_hybrid}')

    sample_mean_z0_hybrid = torch.mean(z_t0_hybrid, dim=0)
    sample_cov_z0_hybrid = torch.cov(z_t0_hybrid.T)

    z0_hybrid_mean_rmse = torch.sqrt(
        MSELoss()(sample_mean_z0_hybrid,
                  artifact['base_distribution'].mean.to(torch.device('cpu'))))
    z0_hybrid_cov_rmse = torch.sqrt(
        MSELoss()(torch.diagonal(sample_cov_z0_hybrid),
                  artifact['base_distribution'].variance.to(torch.device('cpu'))))

    logger.info(f'RMSE between base-dist. mean and sample mean for z(t0) hybrid = {z0_hybrid_mean_rmse}')
    logger.info(f'RMSE between base-dist. variance and sample variance for z(t0) hybrid = {z0_hybrid_cov_rmse}')

    logger.info(f'finished step(6)')
    # sys.exit(-1)
    #
    # # TODO should we check MSE for log also, does it make sense ?
    # mean_log_p_z0_hybrid = torch.mean(artifact['base_distribution']
    #                                   .log_prob(z_t0_hybrid.to(torch.device('cuda:0'))), dim=0).item()
    # logger.info(f'mean log-p(z(t0)) hybrid = {mean_log_p_z0_hybrid}')
    # logger.info(
    #     f'Finished hybrid ode trajectory generation with mse(z(t0)) = {mse_val_z0_hybrid} and mse(log(p(z0))) = '
    #     f'{mse_val_logp_z0_hybrid}')
    #
    # ## Generate the
    # #
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
