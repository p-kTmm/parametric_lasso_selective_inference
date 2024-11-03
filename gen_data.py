import numpy as np


def generate(n, p, beta_vec):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)
        noise = np.random.normal(0, 1)
        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y


def generate_non_normal(n, p, beta_vec):
    X = []
    y = []
    true_y = []

    for i in range(n):
        X.append([])
        yi = 0
        for j in range(p):
            xij = np.random.normal(0, 1)
            X[i].append(xij)
            yi = yi + xij * beta_vec[j]

        true_y.append(yi)

        noise = np.random.normal(0, 1)
        # noise = np.random.laplace(0, 1)
        # noise = skewnorm.rvs(a=10, loc=0, scale=1)
        # noise = np.random.standard_t(20)

        yi = yi + noise
        y.append(yi)

    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y





# =================================================================TRANSFER LEARNING====================================================================#

import numpy as np
from sklearn.linear_model import Lasso

def ind_set(n_vec, k_vec):
    ind_re = []
    for k in k_vec:
        if k == 0:
            ind_re.extend(range(n_vec[0]))
        else:
            start = sum(n_vec[:k])
            end = sum(n_vec[:k+1])
            ind_re.extend(range(start, end))
    return ind_re

def rep_col(x, n):
    return np.tile(x.reshape(-1, 1), n)

def las_kA(X, y, A0, n_vec, alpha):
    p = X.shape[1]
    size_A0 = len(A0)

    if size_A0 > 0:
        ind_kA = ind_set(n_vec, [0] + [a + 1 for a in A0])
        ind_1 = np.arange(n_vec[0])

        y_A = y[ind_kA]
        X_kA = X[ind_kA]

        lasso = Lasso(alpha=alpha, fit_intercept=False).fit(X_kA, y_A)
        w_kA = lasso.coef_
        w_kA[np.abs(w_kA) < alpha] = 0

        residual = y[ind_1] - X[ind_1] @ w_kA
        lasso_delta = Lasso(alpha=alpha, fit_intercept=False)
        lasso_delta.fit(X[ind_1], residual)
        delta_kA = lasso_delta.coef_
        delta_kA[np.abs(delta_kA) < alpha] = 0

        beta_kA = w_kA + delta_kA
    else:
        X1 = X[:n_vec[0]]
        y1 = y[:n_vec[0]]

        lasso = Lasso(alpha=alpha, fit_intercept=False).fit(X1, y1)
        beta_kA = lasso.coef_
        w_kA = None

    return {'beta_kA': beta_kA, 'w_kA': w_kA}

def mse_fun(beta, est, X_test=None):
    est_err = np.sum((beta - est) ** 2)
    pred_err = None
    if X_test is not None:
        pred_err = np.mean((X_test @ (beta - est)) ** 2)
    return {'est_err': est_err, 'pred_err': pred_err}

def Coef_gen(s, h, q=30, size_A0=0, M=10, sig_beta=0.3, sig_delta1=0.3,
             sig_delta2=0.5, p=500, exact=True):
    beta0 = np.concatenate([np.full(s, sig_beta), np.zeros(p - s)])
    W = rep_col(beta0, M)
    W[0, :] -= 2 * sig_beta
    for k in range(M):
        if k < size_A0:
            if exact:
                samp0 = np.random.choice(p, h, replace=False)
                W[samp0, k] += -sig_delta1
            else:
                W[:100, k] += np.random.normal(0, h / 100, 100)
        else:
            if exact:
                samp1 = np.random.choice(p, q, replace=False)
                W[samp1, k] += -sig_delta2
            else:
                W[:100, k] += np.random.normal(0, q / 100, 100)
    return {'W': W, 'beta0': beta0}


def gen_data_transfer(
    n_vec=None, s=16, h=6, q=32, size_A0=12, M=20, sig_beta=0.3,
    sig_delta1=0.3, sig_delta2=0.5, p=500, exact=False
):
    np.random.seed(123)
    # Default n_vec if not provided
    if n_vec is None:
        n0 = 150
        n_vec = [n0] + [100] * M
    
    # Generate coefficients
    coef_all = Coef_gen(
        s=s, h=h, q=q, size_A0=size_A0, M=M, sig_beta=sig_beta,
        sig_delta1=sig_delta1, sig_delta2=sig_delta2, p=p, exact=exact
    )
    B = np.column_stack([coef_all['beta0'], coef_all['W']])
    beta0 = coef_all['beta0']

    # Generate data
    X_list = []
    y_list = []
    y_true_list = []
    for k in range(M + 1):
        X_k = np.random.multivariate_normal(
            mean=np.zeros(p), cov=np.eye(p), size=n_vec[k]
        )
        y_true_k = X_k @ B[:, k]  # True response without noise
        y_k = y_true_k + np.random.normal(0, 1, n_vec[k])  # Add noise
        X_list.append(X_k)
        y_list.append(y_k)
        y_true_list.append(y_true_k)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    y_true = np.concatenate(y_true_list)
    
    return X, y, y_true, beta0
