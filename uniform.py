import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

import parametric_lasso
import gen_data
import util

def run_simulation(num_simulations=1000):
    n = 100
    p = 10
    lamda = 0.05
    beta_vec = [0, 0, 0, 0, 0]  # Under the null hypothesis

    cov = np.identity(n)
    threshold = 20
    p_values = []

    for sim in range(num_simulations):
        X, y, _ = gen_data.generate(n, p, beta_vec)

        clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)
        clf.fit(X, y)
        bh = clf.coef_

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
