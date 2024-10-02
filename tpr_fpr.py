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

def plot_and_save(x_values, y_values, ylabel, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Sample size (n)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def tpr_experiment():
    n_list = [50, 100, 150, 200]

    num_trials = 100  # per repetition
    num_reps = 10  # number of repetitions
    p = 5  # number of features
    lamda = 0.05
    threshold = 20
    alpha = 0.05  # significance level for hypothesis testing
    tpr_values = []  # To store TPR values

    for n in n_list:
        total_correctly_detected = 0  # number of times truly positive features are selected by Lasso
        total_correctly_rejected = 0  # number of times null hypothesis is rejected for truly positive features
        total_trials_with_detections = 0  # number of trials where at least one feature is selected

        for rep in range(num_reps):
            for trial in range(num_trials):
                # Generate data
                beta_vec = [0.25, 0.25] + [0]*(p - 2)
                cov = np.identity(n)
                X, y, _ = gen_data.generate(n, p, beta_vec)

                # Fit Lasso
                clf = linear_model.Lasso(alpha=lamda, fit_intercept=False)
                clf.fit(X, y)
                bh = clf.coef_

                y = y.reshape((n, 1))

                # Construct necessary variables
                A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

                # Check if any features were selected
                if len(A) == 0:
                    continue  # No features selected, skip to next trial

                total_trials_with_detections += 1

                # Number of truly positive features selected by Lasso
                true_positives = [i for i in A if beta_vec[i] != 0]
                num_correctly_detected = len(true_positives)
                total_correctly_detected += num_correctly_detected

                m = len(A)  # Number of hypotheses tested
                alpha_corrected = alpha / m  # Bonferroni correction

                for j_selected in A:
                    etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

                    list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                        X, y, lamda, etaj, n, p, threshold)
                    p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

                    # Check if p-value is less than alpha_corrected
                    if p_value <= alpha_corrected and beta_vec[j_selected] != 0:
                        total_correctly_rejected += 1

        if total_correctly_detected > 0:
            tpr = total_correctly_rejected / total_correctly_detected
        else:
            tpr = 0

        tpr_values.append(tpr)  # Store TPR value
        print(f'n={n}, TPR={tpr:.4f}')

    # Save results to CSV and plot
    save_to_csv(n_list, tpr_values, 'tpr_results.csv')
    plot_and_save(n_list, tpr_values, 'True Positive Rate (TPR)', 'TPR vs Sample Size', 'tpr_plot.png')

def fpr_experiment():
    n_list = [100, 200, 300, 400, 500]

    num_trials = 100  # number of trials
    p = 5
    lamda = 0.05
    threshold = 20
    alpha = 0.05  # significance level for hypothesis testing
    fpr_values = []  # To store FPR values

    for n in n_list:
        total_false_positives_detected = 0
        total_false_positives_rejected = 0
        total_trials_with_detections = 0

        for trial in range(num_trials):
            # Generate data
            beta_vec = [0]*p  # All beta are zero
            cov = np.identity(n)
            X, y, _ = gen_data.generate(n, p, beta_vec)

            # Fit Lasso
            clf = linear_model.Lasso(alpha=lamda, fit_intercept=False)
            clf.fit(X, y)
            bh = clf.coef_

            y = y.reshape((n, 1))

            # Construct necessary variables
            A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

            # Check if any features were selected
            if len(A) == 0:
                continue  # No features selected, skip to next trial

            total_trials_with_detections += 1

            # Number of truly null features selected by Lasso
            false_positives = [i for i in A if beta_vec[i] == 0]
            num_false_positives_detected = len(false_positives)
            total_false_positives_detected += num_false_positives_detected

            m = len(A)  # Number of hypotheses tested
            alpha_corrected = alpha / m  # Bonferroni correction

            for j_selected in A:
                etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)

                list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(
                    X, y, lamda, etaj, n, p, threshold)
                p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

                # Check if p-value is less than alpha_corrected
                if p_value <= alpha_corrected and beta_vec[j_selected] == 0:
                    total_false_positives_rejected += 1

        if total_false_positives_detected > 0:
            fpr = total_false_positives_rejected / total_false_positives_detected
        else:
            fpr = 0

        fpr_values.append(fpr)  # Store FPR value
        print(f'n={n}, FPR={fpr:.4f}')

    # Save results to CSV and plot
    save_to_csv(n_list, fpr_values, 'fpr_results.csv')
    plot_and_save(n_list, fpr_values, 'False Positive Rate (FPR)', 'FPR vs Sample Size', 'fpr_plot.png')

if __name__ == '__main__':
    print('Running TPR experiments...')
    tpr_experiment()
    print('Running FPR experiments...')
    fpr_experiment()
