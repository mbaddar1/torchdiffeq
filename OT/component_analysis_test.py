"""
Test PCA for composing and reconstructing Multivariate Distributions


Comparing patterns of component loadings: Principal
Component Analysis (PCA) versus Independent Component
Analysis (ICA) in analyzing multivariate non-normal data
https://link.springer.com/article/10.3758/s13428-012-0193-1#:~:text=For%20the%20multivariate%20normal%20distribution,ICA%20results%20will%20be%20similar.

Scribe Notes: Multivariate Gaussians and PCA
https://www.cs.princeton.edu/courses/archive/fall10/cos513/notes/2010-11-15.pdf

PCA vs ICA
https://www.quora.com/Are-there-implicit-Gaussian-assumptions-in-the-use-of-PCA-principal-components-analysis
http://compneurosci.com/wiki/images/4/42/Intro_to_PCA_and_ICA.pdf


Independent Component Analysis A Tutorial
https://www.cs.jhu.edu/~ayuille/courses/Stat161-261-Spring14/HyvO00-icatut.pdf

Comparing patterns of component loadings: Principal
Component Analysis (PCA) versus Independent Component
Analysis (ICA) in analyzing multivariate non-normal data
https://link.springer.com/article/10.3758/s13428-012-0193-1
"""
import torch.distributions
from sklearn.decomposition import FastICA
from torch.nn import MSELoss
import pingouin as pg
if __name__ == '__main__':
    D = 4
    N = 10000
    base_dist_cov = torch.diag(torch.distributions.Uniform(0.1, 1).sample(torch.Size([D])))
    A = torch.distributions.Uniform(0.1, 5).sample(torch.Size([D, D]))
    target_dist_cov = torch.matmul(A, A.T)
    base_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=base_dist_cov)
    target_dist = torch.distributions.MultivariateNormal(loc=torch.tensor([0.1, 0.2, -0.8, 0.9]),
                                                         covariance_matrix=target_dist_cov)
    X = base_dist.sample(torch.Size([N]))
    Y = base_dist.sample(torch.Size([N]))
    transformer = FastICA(random_state=0, whiten='unit-variance',max_iter=1000,tol=0.01)
    transformer.fit(Y)
    Y_ICA = torch.tensor(transformer.transform(Y))
    Y_ICA_cov = torch.cov(Y_ICA.T)
    Y_reconstructed = torch.tensor(transformer.inverse_transform(Y_ICA.detach().numpy()))
    print(f'mse sample mean = {MSELoss()(torch.mean(Y,dim=0),torch.mean(Y_reconstructed,dim=0))}')
    print(f'mse sample cov = {MSELoss()(torch.cov(Y.T),torch.cov(Y_reconstructed.T))}')
    print(f'norm test main sample = {pg.multivariate_normality(X=Y.detach().numpy())}')
    print(f'norm test reconstructed sample = {pg.multivariate_normality(X=Y_reconstructed.detach().numpy())}')
    print(f'finished')

