"""
https://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html
https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html#sphx-glr-auto-examples-cluster-plot-linkage-comparison-py
https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
https://cmdlinetips.com/2018/03/pca-example-in-python-with-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
"""
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.model_selection import train_test_split
import random
import numpy as np

from OT.utils import uv_sample

SEEDS = []
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
    # TODO
    """
    1. Experiment 1 -> validate visually
    2. Experiment 2 -> validate visually
    3. Validate 1 and 2 via sink-horn -> David
    """
    N = 1000
    factor = 0.3
    noise = 0.05
    random_state = 0
    shuffle = True
    dataset = "mvn2d"
    transformer_class = "ica"
    kernel = 'rbf'

    # dataset
    print(f'Building dataset : {dataset}')
    if dataset == "circles":
        X, y = make_circles(n_samples=N, factor=factor, noise=noise, random_state=random_state, shuffle=shuffle)
    elif dataset == "moons":
        X, y = make_moons(n_samples=N, noise=noise, random_state=random_state, shuffle=shuffle)
    elif dataset == 'mvn2d':
        A = torch.tensor([[-5, 30.0], [0.0, -10.0]])
        cov_mtx = torch.matmul(A,A.T)
        X = torch.distributions.MultivariateNormal(loc=torch.tensor([-1.0, 1]), covariance_matrix=cov_mtx).sample(
            torch.Size([N])).detach().numpy()
        y = torch.zeros(torch.Size([N])).detach().numpy()
    else:
        raise ValueError(f'Invalid dataset : {dataset}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    """
    Experiment 1 : 
    Y-> Y_PCA -> Y_hat 
    compare Y to Y_hat (visual) 
    """
    print(f'Start Experiment 1 : Y--(ind-decomp)-->  Y_comp --(inv-decomp)--> Y_hat (visual)')
    print(f'Building Transformer : {transformer_class}')
    if transformer_class == 'pca':
        transformer = PCA(whiten=True)
    elif transformer_class == 'kernel_pca':
        transformer = KernelPCA(kernel=kernel, n_components=2, fit_inverse_transform=True)
    elif transformer_class == 'ica':
        transformer = FastICA()
    else:
        raise ValueError(f'Invalid transformer : {transformer_class}')
    print(f'Fitting and Applying transformer {transformer.__class__.__name__}')
    transformer.fit(X_train)
    train_comps = transformer.transform(X_train)
    test_comps = transformer.transform(X_test)
    print(f'Transformer = {transformer}, dataset = {dataset}')
    print(f'Train components Cov = {np.cov(train_comps.T)}')
    print(f'Test components Cov = {np.cov(test_comps.T)}')
    # plot components
    print(f'Plotting components')
    fig, (orig_data_ax, comp_proj_ax) = plt.subplots(
        ncols=2, figsize=(10, 4)
    )

    orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    orig_data_ax.set_ylabel("Feature #1")
    orig_data_ax.set_xlabel("Feature #0")
    orig_data_ax.set_title("Testing data")

    comp_proj_ax.scatter(test_comps[:, 0], test_comps[:, 1], c=y_test)
    comp_proj_ax.set_ylabel("Test component #1")
    comp_proj_ax.set_xlabel("Test component #0")
    comp_proj_ax.set_title(f"Project of test components data\n using {str(transformer)}")
    plt.savefig(f'{dataset}_{str(transformer.__class__.__name__)}_components.jpg')
    plt.clf()
    print(f'Finished plotting components')

    # plot reconstruction
    print(f'Reconstructing from Components')
    print(f'Reconstructing data from components')
    X_reconstructed_indep_comp = transformer.inverse_transform(transformer.transform(X_test))
    fig, (orig_data_ax, comp_back_proj_ax) = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(10, 4)
    )

    orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    orig_data_ax.set_ylabel("Feature #1")
    orig_data_ax.set_xlabel("Feature #0")
    orig_data_ax.set_title("Original test data")

    comp_back_proj_ax.scatter(X_reconstructed_indep_comp[:, 0], X_reconstructed_indep_comp[:, 1], c=y_test)
    comp_proj_ax.set_xlabel("Feature #0")
    comp_back_proj_ax.set_title(f"Reconstruction via {transformer.__class__.__name__}")
    plt.savefig(f'{dataset}_{transformer.__class__.__name__}_reconstruction.jpg')
    plt.clf()
    print(f'Finished Reconstructing')
    print(f'Finished Experiment 1')
    print('---')
    """
    Experiment 2 : 
    Y-> Y_ic (indeop comp) 
    Y-> Y_icq ( indep comp quantiles) 
    Y_icq -- (inv sampling) --> Y_hat
    compare Y to Y_hat (visually) 
    """

    print(f"""Starting Experiment 2 : 
    Y-> Y_ic (indeop comp) 
    Y-> Y_icq ( indep comp quantiles) 
    Y_icq -- (inv sampling) --> Y_hat
    compare Y to Y_hat (visually) """)
    eps = 1e-6
    p_step = 1e-5
    p_values = torch.arange(start=eps, end=1 - eps, step=p_step)
    X_train_icq = torch.quantile(input=torch.tensor(train_comps), q=p_values, dim=0)
    X_ic_qinv = torch.stack([
        uv_sample(Yq=X_train_icq[:, d].reshape(-1), N=N, p_levels=p_values,
                  interp='cubic') for d in range(2)]).T
    m0 = np.cov(X_train.T)
    m1 = torch.mean(torch.tensor(train_comps), dim=0)
    m2 = torch.mean(torch.tensor(X_ic_qinv), dim=0)
    c1 = torch.cov(torch.tensor(train_comps).T)
    c2 = torch.cov(torch.tensor(X_ic_qinv).T)
    X_icq_recons = transformer.inverse_transform(X_ic_qinv.detach().numpy())

    fig, (orig_data_ax, comp_qinv) = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(10, 4)
    )

    orig_data_ax.scatter(train_comps[:, 0], train_comps[:, 1])
    orig_data_ax.set_ylabel("Component #1")
    orig_data_ax.set_xlabel("Component #0")
    orig_data_ax.set_title("Ground-truth Components")

    comp_qinv.scatter(X_ic_qinv[:, 0], X_ic_qinv[:, 1])
    comp_qinv.set_ylabel("Comp #1")
    comp_qinv.set_xlabel("Comp #0")
    comp_qinv.set_title(f"Q-inv components via {transformer.__class__.__name__}")
    plt.savefig(f'{dataset}_{transformer.__class__.__name__}_q_inv_components.jpg')
    plt.clf()

    fig, (orig_data_ax, recons_qinv) = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(10, 4)
    )

    orig_data_ax.scatter(X_train[:, 0], X_train[:, 1])
    orig_data_ax.set_ylabel("Component #1")
    orig_data_ax.set_xlabel("Component #0")
    orig_data_ax.set_title("Ground-truth Data")

    recons_qinv.scatter(X_icq_recons[:, 0], X_icq_recons[:, 1])
    recons_qinv.set_ylabel("Feature #1")
    recons_qinv.set_xlabel("Feature #0")
    recons_qinv.set_title(f"Q-inv recons via {transformer.__class__.__name__}")
    plt.savefig(f'{dataset}_{transformer.__class__.__name__}_q_inv_recons.jpg')
    plt.clf()

    print(f'Finished Reconstructing')
    print(f'Finished Experiment 2')
    """
    After finishing Experiments 1 and 2 => validate via sink-horn 
    """
    pass
