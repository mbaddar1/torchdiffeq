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
    niter = len(loss_curve)
    plt.plot(np.arange(1, niter + 1), loss_curve)
    graph_out_file = os.path.join(args.dir, f'loss_curve_{file_name_without_ext}.png')
    plt.savefig(graph_out_file)
    logger.info(f'Written loss curve graph to {graph_out_file}')
