import argparse
import logging
import os.path
import pickle

import numpy as np
from matplotlib import pyplot as plt

# args
parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, required=True, help='pkl lnode artifacts file')
parser.add_argument('--dir', type=str, required=False, help='graphs dir', default='graphs')
parser.add_argument('--niter', type=int, required=False, default=-1, help='niter to plot from loss curve, '
                                                                          'default = -1 meaning all iters are to be used')
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
if __name__ == '__main__':
    logger.info(f'Args : {args}')
    # 1. Loss Curve Generation
    file_name = os.path.basename(args.input_file)
    file_name_without_ext = os.path.splitext(file_name)[0]
    artifact = pickle.load(open(args.input_file, "rb"))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f"{artifact['trajectory-opt']} CNF Loss Curve - Gaussian {artifact['dim']}d")
    loss_curve = artifact['loss_curve']
    N = len(loss_curve)
    if args.niter > 0:
        N = min(args.niter, N)
    loss_curve = loss_curve[:N]
    # This is the avg loss curve ( i.e. average of last raw losses)
    # FIXME modify the names to be more meaningful
    plt.plot(np.arange(1, N + 1), loss_curve)
    plt.yticks(list(np.arange(0, 3, 0.1)))
    plt.grid()
    graph_out_file = os.path.join(args.dir, f'loss_curve_{file_name_without_ext}.png')
    plt.savefig(graph_out_file)
    logger.info(f'Written loss curve graph to {graph_out_file}')
