#----------------------------------------------ORIGINALLLLL (JUST CHANGE GENERATE DATA) ---------------------------


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
#     num_aux_dataset = 8
#     lamda = 0.05

#     # lamda_w=0.1
#     # lamda_delta=0.05
  
#     sig_beta = 0.3
#     gamma = 0.2
#     threshold = 20
#     p_values = []
#     cov = np.identity(n0)  
  
#     for sim in range(num_simulations):
#         if (sim+1)%250==0 or sim==0:
#           print("----------------Run",sim+1,"----------------",)
#         X_stack, y_stack, n_vec, _ = gen_data.generate_data(p=p, n0=n0, num_aux_dataset=num_aux_dataset, s=s, sig_beta=sig_beta, gamma=gamma)

#         X_0, y_0 = X_stack[:n_vec[0]], y_stack[:n_vec[0]]
#         beta_hat = Lasso(alpha=lamda, fit_intercept=False).fit(X_0, y_0).coef_

#         y_0 = y_0.reshape((-1, 1))
#         M, X_0_M, Mc, X_0_Mc, beta_hat_A = util.construct_A_XA_Ac_XAc_bhA(X_0, beta_hat, n0, p)

#         if len(M) == 0:
#             continue

#         for j_selected in M:
#             etaj, etajTy = util.construct_test_statistic(j_selected, X_0_M, y_0, M)
#             list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
#                 X_0, y_0, lamda, etaj, n0, p, threshold)
#             p_value = util.p_value(
#                 M, beta_hat, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)
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






#-------------------------------------------------------------------------
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

import parametric_lasso
import util
import gen_data

def run_simulation(num_simulations=2000):
    p = 10
    s=0
    n0 = 150
    num_aux_dataset = 8
    lamda = lamda_w = lamda_delta = 0.1

    # lamda_w=0.1
    # lamda_delta=0.05
  
    sig_beta = 0.3
    gamma = 0.2
    threshold = 20
    p_values = []
    cov = np.identity(n0)  
  
    for sim in range(num_simulations):
        if (sim+1)%250==0 or sim==0:
          print("----------------Run",sim+1,"----------------",)
        X_stack, y_stack, n_vec, _ = gen_data.generate_data(p=p, n0=n0, num_aux_dataset=num_aux_dataset, s=s, sig_beta=sig_beta, gamma=gamma)

        beta_hat, w_hat_A, delta_hat_A = gen_data.OracleTransLasso(X_stack, y_stack, n_vec, lamda_w=lamda_w, lamda_delta=lamda_delta)
        X_0, y_0 = X_stack[:n_vec[0]], y_stack[:n_vec[0]]
        X_A, y_A = X_stack[n_vec[0]:], y_stack[n_vec[0]:]

        # #test-------------
        # beta_hat = w_hat_A
        # n0 = num_aux_dataset*100
        # X_0 =X_A
        # y_0 =y_A
        # cov = np.identity(n0) 
        # #-----------------


      
        y_0 = y_0.reshape((-1, 1))
        M, X_0_M, Mc, X_0_Mc, beta_hat_A = util.construct_A_XA_Ac_XAc_bhA(X_0, beta_hat, n0, p)

        if len(M) == 0:
            continue

        for j_selected in M:
            etaj, etajTy = util.construct_test_statistic(j_selected, X_0_M, y_0, M)
            list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                X_0, y_0, lamda, etaj, n0, p, threshold)
            p_value = util.p_value(
                M, beta_hat, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)
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

