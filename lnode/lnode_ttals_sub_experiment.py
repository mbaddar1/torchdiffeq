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
from typing import List

import pingouin as pg
import numpy as np
import torch

from GMSOC.functional_tt_fabrique import orthpoly, Extended_TensorTrain
from examples.models import CNF

# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Global-vars
N_SAMPLES = 4096

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

# parser.add_argument('--t0-index', type=int, required=True)
args = parser.parse_args()
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


def adjust_tensor_to_domain(x: torch.Tensor, domain_stripe: List[float]):
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


def run_tt_als(x: torch.Tensor, t: float, y: torch.Tensor, poly_degree: int, rank: int, test_ratio: float,
               tol: float) -> Extended_TensorTrain:
    N = x.shape[0]
    N_test = int(test_ratio * N)
    N_train = N - N_test
    #
    x = torch.cat(tensors=[x, torch.tensor([t]).repeat(N, 1)], dim=1)
    Dx = x.shape[1]
    Dy = y.shape[1]
    #
    degrees = [poly_degree] * Dx
    ranks = [1] + [rank] * (Dx - 1) + [1]
    order = len(degrees)  # not used, but for debugging only
    domain = [[-1., 1.] for _ in range(Dx)]
    domain_stripe = domain[0]
    op = orthpoly(degrees, domain)

    ETT = Extended_TensorTrain(op, ranks)
    for d_y in range(Dy):
        y_d = y[:, d_y].view(-1, 1)
        # ALS parameters
        reg_coeff = 1e-2
        iterations = 40
        x_domain_adjusted = adjust_tensor_to_domain(x=x, domain_stripe=domain_stripe)
        rule = None
        # rule = tt.Dörfler_Adaptivity(delta = 1e-6,  maxranks = [32]*(n-1), dims = [feature_dim]*n, rankincr = 1)
        ETT.fit(x=x_domain_adjusted.type(torch.float64)[:N_train, :], y=y_d.type(torch.float64)[:N_train, :],
                iterations=iterations, rule=rule, tol=tol,
                verboselevel=1, reg_param=reg_coeff)
        ETT.tt.set_core(Dx - 1)
        train_error = (torch.norm(ETT(x_domain_adjusted.type(torch.float64)[:N_train, :]) -
                                  y_d.type(torch.float64)[:N_train, :]) ** 2 / torch.norm(
            y_d.type(torch.float64)[:N_train, :]) ** 2).item()
        val_error = (torch.norm(ETT(x_domain_adjusted.type(torch.float64)[N_train:, :]) -
                                y_d.type(torch.float64)[N_train:, :]) ** 2 / torch.norm(
            y_d.type(torch.float64)[N_train:, :]) ** 2).item()
        logger.info(f'For d_y  = {d_y} :TT-ALS Relative error on training set = {train_error}')
        logger.info(f'For d_y = {d_y}: TT-ALS Relative error on test set = {val_error}')
        logger.info("======== Finished TT-ALS training ============")
    return ETT


def verify_fit_ETT(ETT_fit: Extended_TensorTrain,cnf_trajectory_model : CNF, target_distribution: torch.distributions.Distribution,
                   base_distribution: torch.distributions.Distribution, t0: float, tN: float)->float:
    """
    In words, this function verify that  : Given a fitted ETT model (TT + Basis) and a fitted CNF trajectory model
    ( cnf_trajectory model) a sample x following a base distribution p_X can be
    normalized from a sample Y following distribution p_Y by solvig ODE over trajectory tN->t0:

        i) x = z(t0) , y = z(tN)
        ii) from z(tN) to z(tN-h) : one explicit RK step ( euler step)
            z(tN) = y ~ target_distribution , sampling
            z(tN-h) = z(tN) + \int_{tN}^{tN-h} (f_TT(z(tN,tN)) dt
            z(tN-h) = z(tN) + (-h) f_TT(z(tN),tN)
        iii) from z(tN-h) to z(t0)
            z(t0) = odeint(z(tN-h) , tN-h -> t0, cnf_trajectory_model)
        iv) calculate log_prob for the base_distribution give z(t0) , should be close to zero
    :return: log_prob_base_dist_z0  log_prob(base_distribution | z0)
    """
    pass


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
    ## 1. Generate samples out of the loaded model and meta-data
    z_tN = artifact['target_distribution'].sample(torch.Size([N_SAMPLES])).to(device)
    logp_diff_tN = torch.zeros(N_SAMPLES, 1).type(torch.float32).to(device)
    t0 = artifact['args']['t0']
    tN = artifact['args']['t1']
    t_vals = torch.tensor(list(np.arange(tN, t0 - 1, -1)))
    logger.info(f'Running CNF trajectory')
    z_t, _ = odeint(func=trajectory_model, y0=(z_tN, logp_diff_tN), t=t_vals)

    # 2. verify the distribution of the generated data
    ## 2.1 Using HZ test
    z_t0_hat = z_t[-1]
    z_t0_np = z_t0_hat.detach().cpu().numpy()
    normality_test_results = pg.multivariate_normality(X=z_t0_np)
    logger.info(f'Normality test results = {normality_test_results}')
    sample_mio = z_t0_hat.mean(0)
    sample_sigma = torch.cov(z_t0_hat.T)
    ## 2.2 verify the parameters of the generated data
    mean_abs_err = torch.norm(sample_mio - artifact['base_distribution'].mean.detach().cpu())
    mean_rel_err = mean_abs_err / torch.norm(
        artifact['target_distribution'].mean)
    cov_abs_err = torch.norm(sample_sigma - artifact['base_distribution'].covariance_matrix.detach().cpu())
    cov_rel_err = cov_abs_err / torch.norm(
        artifact['target_distribution'].covariance_matrix)
    logger.info(f'mean_abs_err = {mean_abs_err}')
    logger.info(f'mean_rel_err = {mean_rel_err}')
    logger.info(f'cov_abs_err = {cov_abs_err}')
    logger.info(f'cov_rel_err = {cov_rel_err}')
    ## 2.3 calculate log-prob
    dist_device = artifact['base_distribution'].mean.device
    log_prob = artifact['base_distribution'].log_prob(z_t0_hat.to(dist_device)).mean(0)
    prob_ = torch.exp(log_prob)
    # benchmark with a sample
    z_t0_benchmark_sample = artifact['base_distribution'].sample(torch.Size([z_t0_hat.shape[1]]))
    log_prob_benchmark = artifact['base_distribution'].log_prob(z_t0_benchmark_sample).mean(0)

    # 3. Run TT-ALS for the RK-step : z(tN) -> z(tN-h)
    t_N = t_vals[0]
    t_N_minus_h = t_vals[args.h_steps]
    h = t_N - t_N_minus_h
    # mind the signs !!
    z_tN = z_t[0]
    z_tN_minus_h = z_t[args.h_steps]
    y = (z_tN_minus_h - z_tN) / (-h)
    x = z_tN

    ETT_fit = run_tt_als(x=x, y=y, t=t_N, poly_degree=args.degree, rank=args.rank, test_ratio=0.2, tol=args.tol)
    logger.info('Fitting TT-ALS for the sub-trajectory finished')
    #
    # # double test with hybrid trajectory
    # ## 1. Explicit Euler step
    # # z(tN-h) = ode_int(z(tN),[tN -> tN-h], f(z(tN),tN))
    # # apply explicit euler one-step
    # # z(tN-h) = z(tN) + \int_{tN}^{tN-h} f(z(tN),tN))
    # # fixme, make domain stripe parametric
    # N = x.shape[0]
    # x = z_tN
    # x = torch.cat([x, torch.tensor([tN]).repeat(N, 1)], dim=1)
    # x_domain_adjusted = adjust_tensor_to_domain(x=x, domain_stripe=[-1, 1])
    # x_domain_adjusted_min = torch.min(x_domain_adjusted, dim=0)
    # x_domain_adjusted_max = torch.max(x_domain_adjusted, dim=0)
    # Dx = x.shape[1]
    # ETT_fit.tt.set_core(Dx - 1)
    # f = ETT_fit(x=x_domain_adjusted)
    # z_tN_minus_h = z_tN - h * f  # euler step, but backward
    # t_vals = torch.tensor([t_N_minus_h, t0])
    # z_t, _ = odeint(func=trajectory_model, y0=(z_tN_minus_h.type(torch.float32), logp_diff_tN), t=t_vals)
    # z_t0_hat = z_t[-1]
    # logger.info(f'Finished hybrid trajectory computation')
