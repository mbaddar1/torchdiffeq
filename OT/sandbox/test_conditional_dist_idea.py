"""
Quantile function for joint distribution
https://math.stackexchange.com/questions/3279346/quantile-function-for-joint-distribution
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_circles
from scipy import stats

from OT.utils import uv_sample

if __name__ == '__main__':
    N = 2000
    factor = 0.3
    noise = 0.05
    random_state = 0
    shuffle = True
    X, y = make_circles(n_samples=N, factor=factor, noise=noise, random_state=random_state, shuffle=shuffle)
    # plot data
    fig, (orig_data_ax) = plt.subplots(
        ncols=1, figsize=(10, 4)
    )

    orig_data_ax.scatter(X[:, 0], X[:, 1], c=y)
    orig_data_ax.set_ylabel("Feature #1")
    orig_data_ax.set_xlabel("Feature #0")
    orig_data_ax.set_title("Data")
    plt.savefig('circle.jpg')
    # Test inv transform on 1 d
    d_idx = 1
    ecdf_ = stats.ecdf(X[:, d_idx])
    ecdf_.cdf.plot()
    # plt.savefig('ecdf.jpg')
    p_values = torch.arange(1e-6, 1 - 1e-6, 1e-5)
    Xq = torch.quantile(torch.tensor(X[:, d_idx]), dim=0, q=p_values)
    X_qinv = uv_sample(Yq=torch.tensor(Xq.reshape(-1)), N=N, p_levels=p_values, interp='cubic')
    ecdf_2 = stats.ecdf(X_qinv.detach().numpy())
    # plt.clf()
    ecdf_2.cdf.plot()
    plt.savefig('ecdf_compare.jpg')
    # make conditional quantile training data
    d_other_idx = 1-d_idx # assume d_idx is 0 or 1
    print(f'finished')
