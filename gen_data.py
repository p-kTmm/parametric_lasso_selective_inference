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
import matplotlib.pyplot as plt

def OracleTransLasso (X, y, n_vec, lamda_w=None, lamda_delta=None):
    n0 = n_vec[0]
    nA = np.sum(n_vec) - n0

    X0 = X[:n0,:]
    XA = X[n0:,:]
    y0 = y[:n0]
    yA = y[n0:]

    p = X.shape[1]

    if lamda_w == None:
        lamda_w = np.sqrt(2 * np.log(p)/nA)
    if lamda_delta == None:
        lamda_delta = np.sqrt(2 * np.log(p)/n0)
    
    w_hat_A = Lasso(alpha = lamda_w, fit_intercept=False).fit(XA, yA).coef_

    delta_hat_A = Lasso(alpha = lamda_delta, fit_intercept=False).fit(X0, y0 - X0@w_hat_A).coef_

    beta_hat = w_hat_A + delta_hat_A

    return beta_hat, w_hat_A, delta_hat_A                                                    

def sse(beta, beta_hat):
    return np.sum((beta - beta_hat) ** 2)

def coef_gen(s=2, M=10, sig_beta=0.3, gamma=0.3, p=20): 
        # Generate beta with s non-zero entries and p-s zero entries
        beta = np.concatenate([np.full(s, sig_beta), np.zeros(p - s)])
        
        # Initialize W by replicating beta across M columns
        W = np.tile(beta, (M, 1)).T
    
        # Add noise to informative models (first size_A0 columns)
        for m in range(M):
            W[:, m] += np.random.normal(0, gamma * sig_beta, p)

        return {'W': W, 'beta': beta}


def generate_data(p=20, n0=150, M=10, s=2, sig_beta=0.3, gamma=0.3):
    n_vec = [n0] + [100] * M
    coef_all = coef_gen(s=s, M=M, sig_beta=sig_beta, gamma=gamma, p=p)
    
    B = np.column_stack([coef_all['beta'], coef_all['W']])
    beta = coef_all['beta']

    X_list = []
    y_list = []

    for k in range(M + 1):
        X_k = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p), size=n_vec[k])
        y_k = X_k @ B[:, k] + np.random.normal(0, 1, n_vec[k])
        X_list.append(X_k)
        y_list.append(y_k)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y, n_vec, beta
