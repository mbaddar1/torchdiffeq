"""
PCA tutorial
https://www.cs.cmu.edu/~elaw/papers/pca.pdf

PCA + Math
https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch18.pdf
https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643?gi=4c8ee2a21591
https://www.analyticsvidhya.com/blog/2021/09/pca-and-its-underlying-mathematical-principles/

"""
import datetime
import pingouin as pg
import numpy as np
import torch.distributions
from sklearn.decomposition import PCA
from torch.nn import MSELoss
import random
from OT.models import Reg
from OT.utils import uv_sample, wasserstein_distance_two_gaussians

#
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
#
if __name__ == '__main__':
    torch_dtype = torch.float64
    torch_device = torch.device('cpu')
    D = 4
    N = 100000
    p_step = 1e-6
    p_levels = torch.arange(0, 1.0 + p_step, p_step, dtype=torch_dtype, device=torch_device)
    niter = 300
    avg_loss_window = 50
    start_train_timestamp = datetime.datetime.now().isoformat()
    # get train data
    target_dist_mean = torch.distributions.Uniform(-5, 5).sample(torch.Size([D])).type(torch_dtype).to(torch_device)
    A = torch.distributions.Uniform(-10.0, 10).sample(torch.Size([D, D])).type(torch_dtype).to(torch_device)
    target_dist_cov = torch.matmul(A, A.T).type(torch_dtype)
    target_dist = torch.distributions.MultivariateNormal(loc=target_dist_mean, covariance_matrix=target_dist_cov)
    print(f'target_dist = {target_dist}')
    Y = target_dist.sample(torch.Size([N])).type(torch_dtype).to(torch_device)
    # PCA
    transformer = PCA(whiten=True)
    Y_indep_comp = torch.tensor(transformer.fit_transform(Y.detach().numpy()), dtype=torch_dtype, device=torch_device)
    print(f'cov for Y_indep_comp used for training {torch.cov(Y_indep_comp.T)}')
    Y_indep_comp_q = torch.quantile(input=Y_indep_comp, q=p_levels, dim=0).type(torch_dtype).to(torch_device)
    # model init
    learning_rate = 0.1
    model = Reg(in_dim=D, out_dim=D, hidden_dim=50, model_type='linear', torch_dtype=torch_dtype,
                torch_device=torch_device, bias=False)
    print(f'model = {model}')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    for i in range(niter):
        optimizer.zero_grad()
        # self reg model
        Y_indep_comp_q_hat = model(Y_indep_comp_q)
        loss = MSELoss()(Y_indep_comp_q_hat, Y_indep_comp_q)
        losses.append(loss.item())
        loss_rolling_avg = np.mean(losses[-avg_loss_window:])
        print(f'i = {i + 1} => loss = {loss.item()}')
        loss.backward()
        optimizer.step()

    # print training summary
    train_summary = dict()
    train_summary['SEED'] = SEED
    train_summary['train_start_timestamp'] = start_train_timestamp
    train_summary['target_dist'] = target_dist
    train_summary['torch_dtype'] = torch_dtype
    train_summary['torch_device'] = torch_device
    train_summary['niter'] = niter
    train_summary['N'] = N
    train_summary['lr'] = learning_rate
    train_summary['model'] = str(model)
    train_summary['latest_loss'] = losses[-1]

    # I) In-Sample Testing
    Y_indep_comp_q_pred_test = model(Y_indep_comp_q)
    Y_indep_comp_qinv_sample = torch.stack([
        uv_sample(Yq=Y_indep_comp_q_pred_test[:, i].reshape(-1), N=N, u_levels=p_levels, u_step=p_step, interp='cubic')
        for i in range(D)]).T.type(torch_dtype)
    print(f'Mean Y_indep_comp_qinv_sample = {torch.mean(Y_indep_comp_qinv_sample, dim=0)}')
    print(f'Covariance reconstruct = {torch.cov(Y_indep_comp_qinv_sample.T)}')

    """
    Reconstruct means 
    1. predict Yq
    2. Apply Q-inv sampling to get the sample
    """
    Y_reconstruct = torch.tensor(transformer.inverse_transform(Y_indep_comp_qinv_sample), dtype=torch_dtype,
                                 device=torch_device)
    mean_recons = torch.mean(Y_reconstruct, dim=0)
    cov_recons = torch.cov(Y_reconstruct.T)

    # WD Q-inv and Direct sampling
    Y_benchmark = target_dist.sample(torch.Size([N]))
    mean_benchmark = torch.mean(Y_benchmark, dim=0)
    cov_benchmark = torch.cov(Y_benchmark.T)
    wd_benchmark = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_benchmark,
                                                      C1=target_dist.covariance_matrix, C2=cov_benchmark)
    wd_reconstruct = wasserstein_distance_two_gaussians(m1=target_dist.mean, m2=mean_recons,
                                                        C1=target_dist.covariance_matrix, C2=cov_recons)
    N_hz = 10000
    perm = torch.randperm(N_hz)
    mvn_hz_benchmark = pg.multivariate_normality(X=Y_benchmark[perm[:N_hz], :].detach().numpy())
    mvn_hz_reconstruct = pg.multivariate_normality(X=Y_reconstruct[perm[:N_hz], :].detach().numpy())
    mean_norm_diff = torch.linalg.norm(mean_benchmark - mean_recons)
    cov_norm_diff = torch.linalg.norm(cov_benchmark - cov_recons)
    mean_benchmark_rmse = torch.sqrt(torch.mean((target_dist.mean - mean_benchmark) ** 2))
    cov_benchmark_rmse = torch.sqrt(torch.mean((target_dist.covariance_matrix - cov_benchmark) ** 2))
    mean_recons_rmse = torch.sqrt(torch.mean((target_dist.mean - mean_recons) ** 2))
    cov_recons_rmse = torch.sqrt(torch.mean((target_dist.covariance_matrix - cov_recons) ** 2))
    print(f'MVN HZ test benchmark = {mvn_hz_benchmark}')
    print(f'MVN HZ test reconstruct = {mvn_hz_reconstruct}')
    print(f'wd benchmark = {wd_benchmark}')
    print(f'wd reconstruct = {wd_reconstruct}')
    train_summary['in_sample'] = dict()

    train_summary['in_sample']['mvn_hz_benchmark'] = mvn_hz_benchmark
    train_summary['in_sample']['mvn_hz_reconstruct'] = mvn_hz_reconstruct
    train_summary['in_sample']['wd_benchmark'] = wd_benchmark
    train_summary['in_sample']['wd_reconstruct'] = wd_reconstruct
    train_summary['in_sample']['mean_norm_diff'] = mean_norm_diff
    train_summary['in_sample']['cov_norm_diff'] = cov_norm_diff
    train_summary['in_sample']['mean_rmse_benchmark'] = mean_benchmark_rmse
    train_summary['in_sample']['cov_rmse_benchmark'] = cov_benchmark_rmse
    train_summary['in_sample']['mean_rmse_recons'] = mean_recons_rmse
    train_summary['in_sample']['cov_rmse_recons'] = cov_recons_rmse
    # TODO out of sample regression testing
    print(f'train-summary:\n {train_summary}')

"""
Experiment log quick dump
====
train-summary:
 {'SEED': 42, 'train_start_timestamp': '2023-10-29T14:26:57.629411', 
 'target_dist': MultivariateNormal(loc: torch.Size([4]), covariance_matrix: torch.Size([4, 4])), 
 'torch_dtype': torch.float64, 'torch_device': device(type='cpu'), 'niter': 300, 'N': 100000, 'lr': 0.1, 
 'model': 'Reg(\n  (models): ModuleList(\n    (0-3): 4 x Linear(in_features=4, out_features=1, bias=False)\n  )\n)',
  'latest_loss': 1.9652227578475105e-05, 'in_sample': {'mvn_hz_benchmark': HZResults(hz=1.069303434075939, 
  pval=0.028179061138068232, normal=False), 'mvn_hz_reconstruct': HZResults(hz=1.0036916144802888,
   pval=0.24311899418526467, normal=True), 'wd_benchmark': 0.010912658933138877, 'wd_reconstruct': 
   0.014619708699446211, 'mean_norm_diff': tensor(0.1417), 'cov_norm_diff': tensor(3.5688), 'mean_rmse_benchmark':
    tensor(0.0394), 'cov_rmse_benchmark': tensor(0.4675), 'mean_rmse_recons': tensor(0.0341), 'cov_rmse_recons': 
    tensor(0.6935)}}
---
 {'train_start_timestamp': '2023-10-29T12:29:38.081113', 'torch_dtype': torch.float64, 'torch_device': 
 device(type='cpu'), 'niter': 30000, 'N': 1000, 'lr': 0.1, 'model': 'Reg(\n  (models): 
 ModuleList(\n    (0-3): 4 x Linear(in_features=4, out_features=1, bias=True)\n  )\n)', 
 'latest_loss': 9.563490995944148e-05}
Mean reconstruct = tensor([ 0.0266, -0.0106,  0.0092,  0.0173])
Covariance reconstruct = tensor([[ 0.9587, -0.0193, -0.0608, -0.0016],
        [-0.0193,  1.0294,  0.0106, -0.0014],
        [-0.0608,  0.0106,  1.0397,  0.0178],
        [-0.0016, -0.0014,  0.0178,  1.0028]])
---
train-summary:
without bias  , linear => O(e-5) error
 {'torch_dtype': torch.float32, 'torch_device': device(type='cpu'), 'niter': 30000, 'N': 1000, 'lr': 0.1, 'model': 
 'Reg(\n  (models): ModuleList(\n    (0-3): 4 x Linear(in_features=4, out_features=1, bias=False)\n  )\n)', 
 'latest_loss': 3.4990331187145784e-05}
 ---
 with bias , linear =: O(e-5) error
 train summary:
  {'torch_dtype': torch.float64, 'torch_device': device(type='cpu'), 'niter': 30000, 'N': 1000, 'lr': 0.1, 'model': 
  'Reg(\n  (models): ModuleList(\n    (0-3): 4 x Linear(in_features=4, out_features=1, bias=True)\n  )\n)',
   'latest_loss': 4.4603766813623496e-05}
"""
