import numpy as np
from sklearn import linear_model
import parametric_lasso
import gen_data
import util


def run_tpr_fpr(n, p, lamda, beta_vec, num_trials=100, threshold=20):
    total_relevant_features = sum([1 for b in beta_vec if b != 0])  # Số đặc trưng quan trọng
    total_irrelevant_features = sum([1 for b in beta_vec if b == 0])  # Số đặc trưng không quan trọng

    TPR_trials = []
    FPR_trials = []

    for trial in range(num_trials):
        X, y, true_y = gen_data.generate(n, p, beta_vec)

        # Huấn luyện mô hình Lasso
        clf = linear_model.Lasso(alpha=lamda, fit_intercept=False)
        clf.fit(X, y)
        bh = clf.coef_

        y = y.reshape((n, 1))

        # Tạo các tập A và Ac
        A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X, bh, n, p)

        if len(A) == 0:
            continue  # Nếu không có đặc trưng nào được chọn, bỏ qua trial này

        correctly_detected = 0
        correctly_rejected = 0
        incorrectly_detected = 0

        cov = np.identity(n)  # Covariance matrix cho p-value và pivot

        for j_selected in A:
            etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
            list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X, y, lamda, etaj, n, p, threshold)
            
            # Tính p-value
            p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

            if beta_vec[j_selected] != 0:  # Đây là đặc trưng quan trọng
                correctly_detected += 1
                if p_value < 0.05:  # Nếu p-value nhỏ hơn 0.05
                    correctly_rejected += 1
            else:
                if p_value < 0.05:  # Đặc trưng không quan trọng nhưng bị chọn
                    incorrectly_detected += 1

        # Tính TPR cho trial này
        if correctly_detected > 0:
            TPR = correctly_rejected / correctly_detected
        else:
            TPR = 0

        # Tính FPR cho trial này
        if total_irrelevant_features > 0:
            FPR = incorrectly_detected / total_irrelevant_features
        else:
            FPR = 0

        TPR_trials.append(TPR)
        FPR_trials.append(FPR)

    # Tính trung bình TPR và FPR sau các trials
    TPR_mean = np.mean(TPR_trials)
    FPR_mean = np.mean(FPR_trials)

    return TPR_mean, FPR_mean


def run_experiment():
    n_values = [100, 200, 300, 400, 500]  # Kích thước mẫu khác nhau
    p = 5
    lamda = 1
    beta_vec = [2, 2, 0, 0, 0]  # Hai đặc trưng quan trọng
    num_trials = 100

    TPR_results = []
    FPR_results = []

    for n in n_values:
        print(f"Running experiments for n = {n}")
        TPR, FPR = run_tpr_fpr(n, p, lamda, beta_vec, num_trials)
        TPR_results.append(TPR)
        FPR_results.append(FPR)
        print(f"n = {n}: Mean TPR = {TPR}, Mean FPR = {FPR}")

    # Lưu kết quả và vẽ biểu đồ
    plot_results(n_values, TPR_results, FPR_results)


def plot_results(n_values, TPR_results, FPR_results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # Vẽ FPR
    plt.subplot(1, 2, 1)
    plt.plot(n_values, FPR_results, 'o-', label='FPR')
    plt.xlabel('Sample size')
    plt.ylabel('FPR')
    plt.title('False Positive Rate')

    # Vẽ TPR
    plt.subplot(1, 2, 2)
    plt.plot(n_values, TPR_results, 'o-', label='TPR', color='green')
    plt.xlabel('Sample size')
    plt.ylabel('TPR')
    plt.title('True Positive Rate')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_experiment()
