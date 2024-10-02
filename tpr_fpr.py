import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import parametric_lasso
import gen_data
import util

def save_to_csv(x_values, y_values, filename):
    df = pd.DataFrame({'Sample size': x_values, 'Value': y_values})
    df.to_csv(filename, index=False)

def plot_boxplot_and_save(x_values, y_values_list, ylabel, title, filename):
    plt.figure(figsize=(8, 6))
    plt.boxplot(y_values_list, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.xticks(range(1, len(x_values) + 1), x_values)
    plt.xlabel('Sample size (n)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def tpr_experiment():
    n_list = [50, 100, 150, 200]
    num_trials = 100
    num_reps = 10
    p = 5
    lamda = 0.1
    threshold = 20
    alpha = 0.05
    tpr_values_list = []

    for n in n_list:
        tpr_values = []
        for rep in range(num_reps):
            for trial in range(num_trials):
                beta_vec = [0.25, 0.25] + [0]*(p - 2)
                cov = np.identity(n)
                X, y, _ = gen_data.generate(n, p, beta_vec)

                clf = linear_model.Lasso(alpha=lamda, fit_intercept=False)
                clf.fit(X, y)
                bh = clf.coef_

                y = y.reshape((n, 1))

                A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

                if len(A) == 0:
                    continue

                total_correctly_detected = 0
                total_correctly_rejected = 0
                true_positives = [i for i in A if beta_vec[i] != 0]
                num_correctly_detected = len(true_positives)
                total_correctly_detected += num_correctly_detected

                m = len(A)
                alpha_corrected = alpha / m

                for j_selected in A:
                    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

                    list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                        X, y, lamda, etaj, n, p, threshold)
                    p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

                    if p_value <= alpha_corrected and beta_vec[j_selected] != 0:
                        total_correctly_rejected += 1

                if total_correctly_detected > 0:
                    tpr = total_correctly_rejected / total_correctly_detected
                else:
                    tpr = 0

                tpr_values.append(tpr)

        tpr_values_list.append(tpr_values)
        print(f'n={n}, TPR={np.mean(tpr_values):.4f}')

    save_to_csv(n_list, [np.mean(tpr) for tpr in tpr_values_list], 'tpr_results.csv')
    plot_boxplot_and_save(n_list, tpr_values_list, 'True Positive Rate (TPR)', 'TPR vs Sample Size', 'tpr_boxplot.png')

def fpr_experiment():
    n_list = [100, 200, 300, 400, 500]
    num_trials = 100
    p = 5
    lamda = 0.1
    threshold = 20
    alpha = 0.05
    fpr_values_list = []

    for n in n_list:
        fpr_values = []
        for trial in range(num_trials):
            beta_vec = [0]*p
            cov = np.identity(n)
            X, y, _ = gen_data.generate(n, p, beta_vec)

            clf = linear_model.Lasso(alpha=lamda, fit_intercept=False)
            clf.fit(X, y)
            bh = clf.coef_

            y = y.reshape((n, 1))

            A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

            if len(A) == 0:
                continue

            total_false_positives_detected = 0
            total_false_positives_rejected = 0
            false_positives = [i for i in A if beta_vec[i] == 0]
            num_false_positives_detected = len(false_positives)
            total_false_positives_detected += num_false_positives_detected

            m = len(A)
            alpha_corrected = alpha / m

            for j_selected in A:
                etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

                list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                    X, y, lamda, etaj, n, p, threshold)
                p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

                if p_value <= alpha_corrected and beta_vec[j_selected] == 0:
                    total_false_positives_rejected += 1

            if total_false_positives_detected > 0:
                fpr = total_false_positives_rejected / total_false_positives_detected
            else:
                fpr = 0

            fpr_values.append(fpr)

        fpr_values_list.append(fpr_values)
        print(f'n={n}, FPR={np.mean(fpr_values):.4f}')

    save_to_csv(n_list, [np.mean(fpr) for fpr in fpr_values_list], 'fpr_results.csv')
    plot_boxplot_and_save(n_list, fpr_values_list, 'False Positive Rate (FPR)', 'FPR vs Sample Size', 'fpr_boxplot.png')

if __name__ == '__main__':
    print('Running TPR experiments...')
    tpr_experiment()
    print('Running FPR experiments...')
    fpr_experiment()
