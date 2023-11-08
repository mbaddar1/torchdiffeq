import numpy as np
import torch.distributions
from hungarian_algorithm import algorithm
from sklearn.linear_model import LinearRegression
from torch.nn import MSELoss


def create_cost_matrix(Y: torch.Tensor, Y_hat: torch.Tensor):
    N = len(Y)
    c = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            c[i, j] =np.abs(Y[i] - Y_hat[j])
    return c


if __name__ == '__main__':
    # H = {
    #     'A': {'#191': 22, '#122': 14, '#173': 120, '#121': 21, '#128': 4, '#104': 51},
    #     'B': {'#191': 19, '#122': 12, '#173': 172, '#121': 21, '#128': 28, '#104': 43},
    #     'C': {'#191': 161, '#122': 122, '#173': 2, '#121': 50, '#128': 128, '#104': 39},
    #     'D': {'#191': 19, '#122': 22, '#173': 90, '#121': 11, '#128': 28, '#104': 4},
    #     'E': {'#191': 1, '#122': 30, '#173': 113, '#121': 14, '#128': 28, '#104': 86},
    #     'F': {'#191': 60, '#122': 70, '#173': 170, '#121': 28, '#128': 68, '#104': 104},
    # }
    # m = algorithm.find_matching(H, matching_type='min', return_type='list')
    # print(m)
    ###
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost)
    c = cost[row_ind, col_ind]
    ###
    base_dist = torch.distributions.Normal(loc=0, scale=1)
    target_dist = torch.distributions.Normal(loc=5, scale=2)
    X = base_dist.sample(torch.Size([20]))
    Y = base_dist.sample(torch.Size([20]))

    X = X.detach().numpy()
    Y = Y.detach().numpy()

    big_itr_count = 100
    for i in range(big_itr_count):
        m = LinearRegression(fit_intercept=True)
        m.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
        Y_hat = m.predict(X.reshape(-1, 1))
        loss = MSELoss()(torch.tensor(Y_hat), torch.tensor(Y))
        print(f'loss at iter {i} = {loss}')
        cst_mtx = create_cost_matrix(Y=Y, Y_hat=Y_hat)
        row_ind, col_ind = linear_sum_assignment(cst_mtx)
        #m = algorithm.find_matching(cst_mtx, matching_type='min', return_type='list')
        assign_cost = cst_mtx[row_ind,col_ind].sum()
        print(f'assign_cost at itr {i} = {assign_cost}')
        # reorder
        Y = Y[row_ind]
        X = X[col_ind]
        print('==')
    mean_ = np.nanmean(Y_hat)
    cov_ = np.std(Y_hat.reshape(-1,1))
    print('--')
