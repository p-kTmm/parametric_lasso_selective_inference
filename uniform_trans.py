import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

import parametric_lasso
import gen_data
import util
import warnings
from sklearn.exceptions import ConvergenceWarning

# Tắt cảnh báo ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_simulation(num_simulations=1000):
    # Parameters
    alpha = 0.1
    p = 200
    # s = 16
    s = 0
    M = 20
    sig_beta = 0.3
    n0 = 150
    n_vec = [n0] + [100] * M
    size_A0 = 12
    h = 6
    q = 2 * s
    sig_delta1 = sig_beta
    sig_delta2 = sig_beta + 0.2
    exact = False
    
    # Define A0 (list of relevant task indices)
    A0 = list(range(size_A0))

    # n = n_vec[0]
    n = sum(n_vec)
    cov = np.identity(n)
    threshold = 20
    p_values = []
    lamda = alpha

    
    for sim in range(num_simulations):
        X, y, y_true, beta0 = gen_data.gen_data_transfer(n_vec=n_vec, s=s, h=h, q=q, size_A0=size_A0, M=M, 
                                 sig_beta=sig_beta, sig_delta1=sig_delta1, 
                                 sig_delta2=sig_delta2, p=p, exact=exact)
        # X, y = X[:n_vec[0]], y[:n_vec[0]]
        # clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)
        # clf.fit(X, y)
        # bh = clf.coef_
        # print(bh)


        res_kA = gen_data.las_kA(X, y, A0=A0, n_vec=n_vec, alpha=alpha)
        bh = res_kA['beta_kA']
        
        y = y.reshape((n, 1))

        A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

        if len(A) == 0:
            continue

        for j_selected in A:
            etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

            list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                X, y, lamda, etaj, n, p, threshold)
            p_value = util.p_value(
                A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

            if p_value:
                p_values.append(p_value)

    # Plot the histogram of p-values
    plt.hist(p_values, bins=20, edgecolor='k', density=True)
    plt.xlabel('p-value')
    plt.ylabel('Density')
    plt.title('Histogram of p-values under the Null Hypothesis')
    plt.savefig('pvalue_histogram.png')
    plt.show()

    print(f"Number of p-values collected: {len(p_values)}")

if __name__ == '__main__':
    run_simulation()
