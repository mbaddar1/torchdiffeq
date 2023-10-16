import numpy as np
import torch
from torch.nn import MSELoss
import numpy as np
from sklearn.decomposition import PCA


def get_comp_based_quantile(X: torch.Tensor):
    pass


class LinReg(torch.nn.Module):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    D = 4
    N = 10000
    niter = 10000
    q = torch.tensor(list(np.arange(0.1, 1, 0.01)), dtype=torch.float32)
    X = torch.distributions.MultivariateNormal(loc=torch.zeros(D), covariance_matrix=torch.eye(D)).sample(
        torch.Size([N]))
    A = torch.tensor([[0.1, -0.2, 0.1, -0.4], [0.1, 0.2, 0.3, 2.4], [0.1, 1.2, 2.3, 0.9], [0.2, 0.2, 0.1, 0.4]])
    cov = torch.matmul(A.T, A)
    Y = torch.distributions.MultivariateNormal(loc=torch.tensor([100.0] * D),
                                               covariance_matrix=cov).sample(
        torch.Size([N]))
    test_cov = torch.cov(Y.T)
    Xq = torch.quantile(input=X, q=q, dim=0)
    Yq = torch.quantile(input=Y, q=q, dim=0)
    Ypca = torch.tensor(PCA().fit_transform(Y.detach().numpy()))
    test_cov_pca = torch.cov(Ypca.T)
    YqPca = torch.quantile(input=Ypca, q=q, dim=0)
    #
    learning_rate = 0.1
    model = LinReg(dim=D)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i in range(niter):
        optimizer.zero_grad()
        Yq_hat = model(Xq)
        loss = MSELoss()(YqPca, Yq_hat)
        print(f'loss = {loss}')
        loss.backward()
        optimizer.step()
    print(model.model.weight)
