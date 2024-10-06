import numpy as np
from sklearn import linear_model
from scipy.stats import norm

import parametric_lasso
import gen_data
import util

def calculate_p_value_no_conditioning(etaTx, std_err):
    # Normalize etaTx with standard error to get z-score
    z_score = etaTx / std_err
    
    # Compute two-tailed p-value based on the z-score
    p_value = 2 * min(1 - norm.cdf(z_score), norm.cdf(z_score))
    return p_value

def estimate_standard_error(XA, eta_j):
    # Reshape eta_j to match the dimensions of the active set XA
    eta_j = eta_j.reshape(-1, 1)  # Convert to column vector if necessary

    # Standard error estimation using the covariance structure of XA
    cov_matrix = np.linalg.inv(np.dot(XA.T, XA))  # Covariance matrix of the active set XA
    std_err = np.sqrt(np.dot(np.dot(eta_j.T, cov_matrix), eta_j))
    return std_err[0, 0]  # Extract scalar from the result

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
        # Calculate eta_j for the active set, not the full dataset
        etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
        
        # Ensure eta_j has the correct shape, matching the number of selected features
        etaj = etaj[:len(A)]  # Only use the part of eta_j that corresponds to the active set

        # Estimate the standard error for the test statistic
        std_err = estimate_standard_error(XA, etaj)

        # Calculate p-value without conditioning using a normal distribution
        p_value_no_conditioning = calculate_p_value_no_conditioning(etajTy, std_err)

        print('Feature', j_selected + 1, ' True Beta:', beta_vec[j_selected], 
              ' Unconditioned p-value:', '{:.4f}'.format(p_value_no_conditioning))
        print("==========")

if __name__ == '__main__':
    run()
