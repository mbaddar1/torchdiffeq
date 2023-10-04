"""
This script is to train Vanilla-CNF model as baseline and LNODE model as a Low-Rank approximation to accelerate
 the training for High Dimensional Distributions. The script is based on the original CNF script here
 https://github.com/rtqichen/torchdiffeq/blob/master/examples/cnf.py

Examples for running via command line
=======================================

1) Train a Vanilla-CNF
export PYTHONPATH=$PYTHONPATH:.

2)Train a LNODE model [TBD]

Sample Running command line CMD line
/home/mbaddar/mbaddar/phd/torchdiffeq/venv310/bin/python /home/mbaddar/mbaddar/phd/torchdiffeq/lnode/lnode.py --niters 1000 --distribution gauss4d --t0 0 --t1 10 --trajectory-opt vanilla

"""
import logging
import os
import argparse
import glob
import pickle
import time
from datetime import datetime
from typing import Union, Any, List

from PIL import Image
import numpy as np
import matplotlib
from torch import Tensor

from examples.models import CNF
from utils import domain_adjust, is_domain_adjusted, vanilla_cnf_optimize_step, get_ETTs, hybrid_cnf_optimize_step, \
    validate_hybrid_trained_models, validate_vanilla_cnf_trained_models

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pingouin as pg
from sklearn.datasets import make_circles
from torch.distributions import MultivariateNormal
import torch
import torch.optim as optim
import random

# SEED
SEED = 41
torch.manual_seed(SEED)
random.seed(SEED)

# Global Variables
DIM_DIST_MAP = {'circles': 2, 'gauss3d': 3, 'gauss4d': 4, 'gauss6d': 6, 'gauss10d': 10}
TOL = 1e-4
# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# args
parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=float, required=True)
parser.add_argument('--t1', type=float, required=True)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--max_iters', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
# LNODE
parser.add_argument('--distribution', type=str, choices=['circles', 'gauss3d', 'gauss4d', 'gauss6d', 'gauss10d'])
parser.add_argument('--trajectory-opt', type=str, choices=['vanilla', 'hybrid'])
parser.add_argument('--itr-threshold', type=int, help='iteration switch threshold')

# parser.add_argument('--domain-adjust', default=True, action='store_true',
#                     help='Adjust domain of target random variable to a predefined range for Legendre polynomial and '
#                          'similar basis functions when using TT-ALS in Hybrid trajectory generation')
# pass a list as cmd args
# https://stackoverflow.com/a/32763023/5937273
parser.add_argument('--domain_stripe', nargs='*', type=float, default=[-1, 1],
                    help='domain to adjust target R.V to')
parser.add_argument('--h', type=float, help='h for the TT-RK step')
# parser.add_argument('--tol', type=float, help='tolerance')
args = parser.parse_args()
# assertion for args
if args.trajectory_opt == 'hybrid' and (args.itr_threshold is None or args.itr_threshold < 1):
    raise ValueError(f'if arg.trajectory_opt == hybrid then args.itr_threshold must be an integer >= 1')
domain_stripe = args.domain_stripe
assert isinstance(domain_stripe, list) and len(domain_stripe) == 2 and domain_stripe[0] < domain_stripe[1], \
    f"domain must be of type list, len(list) = 2 and domain[0] < domain[1] , however we got domain = {domain_stripe}"
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# moved to examples.models.trace_df_dz for more modularity
# def trace_df_dz(f, z):
#     """Calculates the trace of the Jacobian df/dz.
#     Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
#     """
#     sum_diag = 0.
#     for i in range(z.shape[1]):
#         sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
#
#     return sum_diag.contiguous()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_distribution(distribution_name: str) -> Union[torch.distributions.Distribution, str]:
    dist = None
    if distribution_name == 'circles':
        return "circles"
    if distribution_name == 'gauss3d':
        loc = torch.tensor([-2.0, 1.0, -5.0])
        scale = torch.diag(torch.tensor([0.5, 0.2, 0.6]))
        dist = MultivariateNormal(loc, scale)
    elif distribution_name == 'gauss4d':
        loc = torch.tensor([-0.1, 0.2, -0.4, 0.4])
        scale = torch.diag(torch.tensor([0.15, 0.01, 0.19, 0.08]))
        dist = MultivariateNormal(loc, scale)
    elif distribution_name == 'gauss6d':
        loc = torch.tensor([-0.8296, 0.0949, -0.4201, 0.0792, -0.9627, -0.7377])
        scale = torch.diag(torch.tensor([0.2020, 0.2895, 0.7083, 0.8840, 0.8146, 0.3650]))
        dist = MultivariateNormal(loc, scale)
    elif distribution_name == 'gauss10d':
        loc = torch.tensor([0.0746, -0.0230, -0.0722, 0.0909, -0.0493, 0.0624])
        scale = torch.diag(torch.tensor([0.1142, 0.0575, 0.1741, 0.1397, 0.1957, 0.1220]))
        dist = MultivariateNormal(loc, scale)
    return dist


def get_batch(num_samples: int, distribution_name: str) -> tuple[Tensor | Any, Any]:
    if distribution_name == 'circles':
        points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
        x = torch.tensor(points).type(torch.float32).to(device)
    else:
        distribution = get_distribution(distribution_name)
        x = distribution.sample(torch.Size([num_samples])).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return x, logp_diff_t1


if __name__ == '__main__':
    # dump args
    logger.info(f'Args:\n{args}')
    # Configs
    # device = torch.device('cuda:' + str(args.gpu)
    #                       if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    in_out_dim = DIM_DIST_MAP[args.distribution]
    base_distribution = MultivariateNormal(loc=torch.zeros(in_out_dim).to(device),
                                           covariance_matrix=0.1 * torch.eye(in_out_dim).to(device))
    target_distribution = get_distribution(args.distribution)
    time_stamp = datetime.now().isoformat()
    # -------
    # Model
    nn_cnf_func = CNF(in_out_dim=in_out_dim, hidden_dim=args.hidden_dim, width=args.width, device=device)
    ETT_fits = get_ETTs(D_in=in_out_dim + 1, D_out=in_out_dim, rank=6, domain_stripe=[-1, 1], poly_degree=6,
                        device=device)
    # optimizer
    optimizer = optim.Adam(nn_cnf_func.parameters(), lr=args.lr)
    # Base distribution
    p_z0 = base_distribution
    loss_meter = RunningAverageMeter()
    loss_running_avg_curve = []
    loss_raw_curve = []

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            nn_cnf_func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        itr_var = 0
        start_time = time.time()
        for itr in range(1, args.max_iters + 1):
            itr_var = itr
            optimizer.zero_grad()
            x, logp_diff_t1 = get_batch(num_samples=args.num_samples, distribution_name=args.distribution)

            if args.trajectory_opt == "vanilla":
                loss = vanilla_cnf_optimize_step(optimizer=optimizer, nn_cnf_model=nn_cnf_func, x=x, t0=args.t0,
                                                 tN=args.t1, logp_diff_tN=logp_diff_t1, adjoint=args.adjoint,
                                                 p_z0=p_z0, device=device)
            elif args.trajectory_opt == "hybrid":
                loss = hybrid_cnf_optimize_step(optimizer=optimizer, nn_cnf_model=nn_cnf_func, ETT_fits=ETT_fits, x=x,
                                                t0=args.t0,
                                                tN=args.t1, logp_ztN=logp_diff_t1, adjoint=args.adjoint, p_z0=p_z0,
                                                device=device, h=args.h, domain_stripe=[-1, 1], itr=itr,
                                                itr_threshold=args.itr_threshold)
            else:
                raise ValueError(f"Unknown trajectory optimization method = {args.trajectory_opt}")
            # calculate wd and hz losses

            loss_raw_curve.append(loss.item())
            if itr > 10:
                my_avg = np.nanmean(loss_raw_curve[-10:])
            else:
                my_avg = np.nanmean(loss_raw_curve)
            loss_running_avg_curve.append(my_avg)
            # if my_avg < 1e-5:
            #     break
            validation_results = None
            if args.trajectory_opt == 'vanilla':
                validation_results = validate_vanilla_cnf_trained_models(nn_cnf_model=nn_cnf_func,
                                                                         base_distribution=base_distribution,
                                                                         target_distribution=target_distribution,
                                                                         t0=args.t0,
                                                                         tN=args.t1,
                                                                         num_samples=args.num_samples,
                                                                         adjoint=args.adjoint)
            elif args.trajectory_opt == 'hybrid':
                validation_results = validate_hybrid_trained_models(nn_cnf_model=nn_cnf_func, ETT_fits=ETT_fits,
                                                                    base_distribution=base_distribution,
                                                                    target_distribution=target_distribution, t0=args.t0,
                                                                    tN=args.t1,
                                                                    num_samples=args.num_samples, adjoint=args.adjoint,
                                                                    h=args.h,
                                                                    domain_stripe=args.domain_stripe, itr=itr,
                                                                    itr_threshold=args.itr_threshold)
            # loss_meter.update(my_avg)

            print('Iter: {}, running avg loss: {:.4f}, raw loss = {:.4f}'.format(itr, my_avg, loss.item()))
            print(f'Iter {itr} , validation results = {validation_results}')
        end_time = time.time()
        logger.info(f'Running time in seconds = {(end_time - start_time)}')
        logger.info('Finished training, dumping experiment results')
        # validate trained models
        validation_results = None
        # if args.trajectory_opt == 'vanilla':
        #     validation_results = validate_vanilla_cnf_trained_models(nn_cnf_model=nn_cnf_func,
        #                                                              base_distribution=base_distribution,
        #                                                              target_distribution=target_distribution,
        #                                                              t0=args.t0,
        #                                                              tN=args.t1,
        #                                                              num_samples=args.num_samples, adjoint=args.adjoint)
        # elif args.trajectory_opt == 'hybrid':
        #     validation_results = validate_hybrid_trained_models(nn_cnf_model=nn_cnf_func, ETT_fits=ETT_fits,
        #                                                         base_distribution=base_distribution,
        #                                                         target_distribution=target_distribution, t0=args.t0,
        #                                                         tN=args.t1,
        #                                                         num_samples=args.num_samples, adjoint=args.adjoint,
        #                                                         h=args.h,
        #                                                         domain_stripe=args.domain_stripe)
        logger.info(f'Validation results for train model using the {args.trajectory_opt} method = {validation_results}')
        train_artifacts = dict()
        train_artifacts['trajectory-opt'] = args.trajectory_opt
        train_artifacts['base_distribution'] = base_distribution

        train_artifacts['dim'] = in_out_dim
        train_artifacts['target_distribution'] = target_distribution
        train_artifacts['max_iters'] = args.max_iters
        train_artifacts['niters'] = itr_var
        train_artifacts['loss'] = loss_meter.avg
        train_artifacts['runtime_in_seconds'] = end_time - start_time
        train_artifacts['args'] = vars(args)
        train_artifacts['model'] = nn_cnf_func.state_dict()
        train_artifacts['loss_curve'] = loss_running_avg_curve
        train_artifacts['loss_curve_raw'] = loss_raw_curve
        # results['domain_adjusted'] = args.domain_adjust
        # results['domain'] = args.domain
        artifact_version_name = (f'{time_stamp}_dist_{target_distribution.__class__.__name__}_d_'
                                 f'{in_out_dim}_niters_{itr_var}_maxiters_{args.max_iters}')
        pickle.dump(obj=train_artifacts,
                    file=open(f'artifacts/{args.trajectory_opt}_{artifact_version_name}.pkl', "wb"))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': nn_cnf_func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t_samples, _ = odeint(
                nn_cnf_func,
                (z_t0, logp_diff_t0),
                torch.tensor(np.linspace(args.t0, args.t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

            z_t_density, logp_diff_t = odeint(
                nn_cnf_func,
                (z_t1, logp_diff_t1),
                torch.tensor(np.linspace(args.t1, args.t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(args.t0, args.t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))

        """
        Experiments Documentation 
        
        Experiment instance
        =====
        Tried to domain adjust x ( target R.V) before running CNF, didn't work
        what happened ?
        
        """

"""
Parameter tracking

for Hybrid
--max_iters
200
--distribution
gauss4d
--t0
0
--t1
10
--trajectory-opt
hybrid
--h
2
--itr-threshold
100
{'hz_loss': 0.05944538011095711, 'wd': 0.008283842113765308}
-----
--max_iters 200 --distribution gauss4d --t0 0 --t1 10 --trajectory-opt vanilla --h 2 --itr-threshold 100
{'hz_loss': 0.8336306754438337, 'wd': 0.0017414402655022247}
HZResults(hz=0.9767139641313509, pval=0.16631441140867886, normal=True)
"""
