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
python3 lnode/lnode_ttals_sub_experiment.py --artifact "lnode/artifacts/vanilla_2023-07-20T13:36:11.559464_dist_MultivariateNormal_d_4_niters_1000.pkl" --trajectory-opt "vanilla" --device "cpu" --rank 2 --degree 3 --h-steps 3
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


def run_tt_als(x: torch.Tensor, t: float, y: torch.Tensor, poly_degree: int, rank: int, test_ratio: float):
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
    order = len(degrees) # not used, but for debugging only
    domain = [[-1., 1.] for _ in range(Dx)]
    domain_stripe = domain[0]
    op = orthpoly(degrees, domain)

    ETT = Extended_TensorTrain(op, ranks)
    for d_y in range(Dy):
        y_d = y[:, d_y].view(-1, 1)
        # ALS parameters
        reg_coeff = 1e-2
        iterations = 40
        tol = 1e-6
        x_domain = adjust_tensor_to_domain(x=x, domain_stripe=domain_stripe)
        rule = None
        # rule = tt.Dörfler_Adaptivity(delta = 1e-6,  maxranks = [32]*(n-1), dims = [feature_dim]*n, rankincr = 1)
        ETT.fit(x=x_domain.type(torch.float64)[:N_train, :], y=y_d.type(torch.float64)[:N_train, :],
                iterations=iterations, rule=rule, tol=tol,
                verboselevel=1, reg_param=reg_coeff)
        ETT.tt.set_core(Dx - 1)
        train_error = (torch.norm(ETT(x_domain.type(torch.float64)[:N_train, :]) -
                                  y_d.type(torch.float64)[:N_train, :]) ** 2 / torch.norm(
            y_d.type(torch.float64)[:N_train, :]) ** 2).item()
        val_error = (torch.norm(ETT(x_domain.type(torch.float64)[N_train:, :]) -
                                y_d.type(torch.float64)[N_train:, :]) ** 2 / torch.norm(
            y_d.type(torch.float64)[N_train:, :]) ** 2).item()
        logger.info(f'For d_y  = {d_y} :TT-ALS Relative error on training set = {train_error}')
        logger.info(f'For d_y = {d_y}: TT-ALS Relative error on test set = {val_error}')
        # print('relative error on validation set: ', val_error)
        print("========================================================")


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
    z0 = artifact['base_distribution'].sample(torch.Size([N_SAMPLES])).to(device)
    logp_diff_t0 = torch.zeros(N_SAMPLES, 1).type(torch.float32).to(device)
    t0 = artifact['args']['t0']
    tN = artifact['args']['t1']
    t_vals = torch.tensor(list(np.arange(t0, tN + 1, 1)))
    logger.info(f'Running CNF trajectory')
    z_t, _ = odeint(func=trajectory_model, y0=(z0, logp_diff_t0), t=t_vals)

    ##2. verify parameters of generated data
    z_tN = z_t[-1]
    z_tN_np = z_tN.detach().cpu().numpy()
    normality_test_results = pg.multivariate_normality(X=z_tN_np)
    logger.info(f'Normality test results = {normality_test_results}')
    sample_mio = z_tN.mean(0)
    sample_sigma = torch.cov(z_tN.T)

    mean_abs_err = torch.norm(sample_mio.detach().cpu() - artifact['target_distribution'].mean)
    mean_rel_err = mean_abs_err / torch.norm(
        artifact['target_distribution'].mean)
    cov_abs_err = torch.norm(sample_sigma.detach().cpu() - artifact['target_distribution'].covariance_matrix)
    cov_rel_err = cov_abs_err / torch.norm(
        artifact['target_distribution'].covariance_matrix)
    logger.info(f'mean_abs_err = {mean_abs_err}')
    logger.info(f'mean_rel_err = {mean_rel_err}')
    logger.info(f'cov_abs_err = {cov_abs_err}')
    logger.info(f'cov_rel_err = {cov_rel_err}')
    # Run TT-ALS with z(t_0) and z(t_1) where t_0 = 0 and t_1  =1
    t_N = t_vals[-1]
    t_N_minus_h = t_vals[-1-args.h_steps]
    x = z_t[-1-args.h_steps]
    y = z_tN
    #
    run_tt_als(x=x, y=y, t=t_N_minus_h, poly_degree=args.degree, rank=args.rank, test_ratio=0.2)
    logger.info('Sub-experiment finished')
