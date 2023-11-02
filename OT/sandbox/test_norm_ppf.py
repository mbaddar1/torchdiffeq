from scipy.stats import norm

if __name__ == '__main__':
    p = [0.1,0.2]
    mean = -2
    scale = 0.2
    for i in range(10):
        q = norm.ppf(q=p, loc=mean, scale=scale)
        p = norm.cdf(x=q, loc=mean, scale=scale)
        print(f'x = {q},p = {p}')
