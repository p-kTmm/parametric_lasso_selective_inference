# import numpy as np
# from sklearn.linear_model import Lasso
# import matplotlib.pyplot as plt

# import parametric_lasso
# import util
# import gen_data

# def run_simulation(num_simulations=1000):
#     p = 10
#     s=0
#     n0 = 150
#     M = 6
#     lamda = 0.05
#     sig_beta = 0.3
#     gamma = 0.3
#     threshold = 20
#     p_values = []
#     cov = np.identity(n0)    
#     for sim in range(num_simulations):
#         X, y, n_vec, _ = gen_data.generate_data(p=p, n0=n0, M=M, s=s, sig_beta=sig_beta, gamma=gamma)
#         X, y = X[:n_vec[0]], y[:n_vec[0]]
#         bh = Lasso(alpha=lamda, fit_intercept=False).fit(X, y).coef_
#         y = y.reshape((-1, 1))
#         A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n0, p)

#         if len(A) == 0:
#             continue

#         for j_selected in A:
#             etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
#             list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
#                 X, y, lamda, etaj, n0, p, threshold)
#             p_value = util.p_value(
#                 A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)
#             if p_value:
#                 p_values.append(p_value)
                
#     plt.hist(p_values, bins=20, edgecolor='k', density=True)
#     plt.xlabel('p-value')
#     plt.ylabel('Density')
#     plt.title('Histogram of p-values under the Null Hypothesis')
#     plt.savefig('pvalue_histogram.png')
#     plt.show()

#     print(f"Number of p-values collected: {len(p_values)}")

# if __name__ == '__main__':
#     run_simulation()


import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

import parametric_lasso
import util
import gen_data

def run_simulation(num_simulations=1000):
    p = 10
    s=0
    n0 = 150
    M = 6
    lamda = 0.05
    sig_beta = 0.3
    gamma = 0.3
    threshold = 20
    p_values = []
    cov = np.identity(n0)    
    for sim in range(num_simulations):
        X, y, n_vec, _ = gen_data.generate_data(p=p, n0=n0, M=M, s=s, sig_beta=sig_beta, gamma=gamma)

        bh = gen_data.OracleTransLasso(X, y, n_vec)
        X, y = X[:n_vec[0]], y[:n_vec[0]]
        y = y.reshape((-1, 1))
        A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n0, p)

        if len(A) == 0:
            continue

        for j_selected in A:
            etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
            list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                X, y, lamda, etaj, n0, p, threshold)
            p_value = util.p_value(
                A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)
            if p_value:
                p_values.append(p_value)
                
    plt.hist(p_values, bins=20, edgecolor='k', density=True)
    plt.xlabel('p-value')
    plt.ylabel('Density')
    plt.title('Histogram of p-values under the Null Hypothesis')
    plt.savefig('pvalue_histogram.png')
    plt.show()

    print(f"Number of p-values collected: {len(p_values)}")

if __name__ == '__main__':
    run_simulation()

