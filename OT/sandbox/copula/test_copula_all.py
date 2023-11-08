"""
To Read
Good articles

Emperical Copula
https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.110/Seminar__Copulas_and_Applications_WS2021/Talk_7_-_Non-parametric_estimation_of_Copulas.pdf
Gaussian Capula with Non-Gaussian Margins
https://stats.stackexchange.com/a/423186

Copula Lecture Notes
https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf

Generating Synthetic Multivariate Data with Copulas
https://towardsdatascience.com/generating-synthetic-multivariate-data-with-copulas-edd1c4afa1bb

Improved Inference of Gaussian Mixture Copula Model for Clustering and Reproducibility Analysis using Automatic Differentiation
https://www.sciencedirect.com/science/article/abs/pii/S2452306221001040

Improved Inference of Gaussian Mixture Copula Model for Clustering and
Reproducibility Analysis using Automatic Differentiation
https://arxiv.org/pdf/2010.14359.pdf


Vine copulas for mixed data : multi-view clustering for mixed data beyond meta-Gaussian dependencies
https://link.springer.com/article/10.1007/s10994-016-5624-2

Copulas as High-Dimensional Generative Models:
Vine Copula Autoencoders
https://serval.unil.ch/resource/serval:BIB_606AA27EDB59.P001/REF.pdf

Clustering Using Student t Mixture Copulas
https://link.springer.com/article/10.1007/s42979-021-00503-0

Good docs
Lecture 12: Copula
https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf

Copula and multivariate dependencies (Different Shapes)
https://risk-engineering.org/static/PDF/slides-copula.pdf

Copulas as High-Dimensional Generative Models: Vine Copula Autoencoders
https://arxiv.org/abs/1906.05423

Non Parametric Copula
https://copulae.readthedocs.io/en/latest/api_reference/copulae/empirical/index.html
https://github.com/maximenc/pycop#Non-parametric-TDC
https://openturns.github.io/openturns/latest/auto_data_analysis/estimate_dependency_and_copulas/plot_estimate_non_parametric_copula.html
Gaussian copula and multivariate normal distribution
https://www.adrian.idv.hk/2019-01-24-copula/
Pair-copula constructions of multivariate
copulas
https://mediatum.ub.tum.de/doc/1079253/651951.pdf

Empirical Copula¶
https://copulae.readthedocs.io/en/latest/api_reference/copulae/empirical/index.html#:~:text=The%20empirical%20copula%2C%20being%20a,%E2%88%88%5B0%2C1%5Dd

https://sdv.dev/Copulas/
https://twiecki.io/blog/2018/05/03/copulas/
https://mediatum.ub.tum.de/doc/1519928/1519928.pdf
https://cran.r-project.org/web/packages/univariateML/vignettes/copula.html
https://mediatum.ub.tum.de/doc/1079253/651951.pdf
https://stats.stackexchange.com/questions/123698/bivariate-sampling-for-distribution-expressed-in-sklars-copula-theorem
https://hal.science/hal-00780082/document
Quantile Regression Via Copula
https://mediatum.ub.tum.de/doc/1360907/766045.pdf
https://www.kaggle.com/code/liamhealy/copulas-in-python
statmodels implementation
https://www.statsmodels.org/dev/examples/notebooks/generated/copula.html

Gaussian Copula and Gaussian Assumption
https://analystprep.com/study-notes/frm/part-2/market-risk-measurement-and-management/financial-correlation-modeling-bottom-up-approaches/#:~:text=A%20Gaussian%20copula%20maps%20the,still%20preserving%20their%20marginal%20distributions.
https://stats.stackexchange.com/questions/479200/when-we-use-gaussian-copula-are-we-implying-that-the-underlying-marginals-are-g

Monte Carlo Methods 3: Vine Copula Methods
https://lewiscoleblog.com/monte-carlo-methods-3#Inverse-Transform-Method

Simulating Dependent Random Variables Using Copulas
https://de.mathworks.com/help/stats/simulating-dependent-random-variables-using-copulas.html

Monte Carlo simulation
https://en.wikipedia.org/wiki/Copula_(probability_theory)#Expectation_for_copula_models_and_Monte_Carlo_integration
----
Already Read
https://en.wikipedia.org/wiki/Copula_(probability_theory)

Regression Copula
Bayesian Inference for Regression Copulas
https://arxiv.org/pdf/1907.04529.pdf 
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from copulas.bivariate import Bivariate, CopulaTypes
from copulas.datasets import sample_trivariate_xyz
from copulas.visualization import scatter_2d, scatter_3d
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import compare_2d, compare_3d
from sklearn.datasets import make_circles

copula = GaussianMultivariate()
if __name__ == '__main__':
    # https://sdv.dev/Copulas/tutorials/00_Quickstart.html
    # data = sample_trivariate_xyz()
    X, y = make_circles(n_samples=2000, shuffle=True, noise=0.05, random_state=0,factor=0.3)
    X_df = pd.DataFrame(X)
    scatter_2d(X_df)
    plt.savefig("data.jpg")
    plt.clf()
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.savefig('cdata.jpg')
    fig, (orig_data_ax) = plt.subplots(
        ncols=1, figsize=(10, 10)
    )
    orig_data_ax.scatter(X[:, 0], X[:, 1], c=y)
    orig_data_ax.set_ylabel("Feature #1")
    orig_data_ax.set_xlabel("Feature #0")
    orig_data_ax.set_title("Testing data")
    # sys.exit(-1)
    # copula = GaussianMultivariate()
    copula = Bivariate(copula_type=CopulaTypes.GUMBEL)
    copula.fit(X)
    check_ = copula.check_fit()
    num_samples = 1000
    synthetic_data = copula.sample(num_samples)
    print(synthetic_data)
    plt.clf()
    compare_2d(X_df, pd.DataFrame(synthetic_data))
    # plt.scatter(synthetic_data[:,0],synthetic_data[:,1],synthetic_data[:,2])
    plt.savefig('compare.jpg')
