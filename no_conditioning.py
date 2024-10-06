import numpy as np
from sklearn import linear_model
from scipy.stats import norm

import parametric_lasso
import gen_data
import util

def calculate_p_value_no_conditioning(etaTx):
    # compute two-tailed p-value
    p_value = 2 * min(1 - norm.cdf(etaTx), norm.cdf(etaTx))
    return p_value

def run():
    n = 100
    p = 5
    lamda = 0.05
    beta_vec = [1, 1, 0, 0, 0]

    cov = np.identity(n)
    threshold = 20

    X, y, _ = gen_data.generate(n, p, beta_vec)

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)
    clf.fit(X, y)
    bh = clf.coef_

    y = y.reshape((n, 1))

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

    if len(A) == 0:
        return None

    for j_selected in A:
        # Calculate eta_j and eta_j^T y
        etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
        
        # Here we calculate p-value without conditioning using a normal distribution
        p_value_no_conditioning = calculate_p_value_no_conditioning(etajTy)

        print('Feature', j_selected + 1, ' True Beta:', beta_vec[j_selected], 
              ' Unconditioned p-value:', '{:.4f}'.format(p_value_no_conditioning))
        print("==========")

if __name__ == '__main__':
    run()
