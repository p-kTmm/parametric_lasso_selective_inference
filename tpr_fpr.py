import numpy as np
from sklearn import linear_model
import parametric_lasso
import gen_data
import util
import matplotlib.pyplot as plt


def run_tpr_fpr(n, p, lamda, beta_vec, num_trials=100, threshold=20, mode='TPR'):
    """Chạy tính TPR hoặc FPR dựa trên chế độ ('TPR' hoặc 'FPR')"""
    if mode == 'TPR':
        total_relevant_features = sum([1 for b in beta_vec if b != 0])  # Số đặc trưng quan trọng
    else:
        total_relevant_features = sum([1 for b in beta_vec if b == 0])  # Số đặc trưng không quan trọng (FPR mode)
    
    TPR_trials = []

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

        cov = np.identity(n)  # Covariance matrix cho p-value và pivot

        for j_selected in A:
            etaj, etajTy = util.construct_test_statistic(j_selected, XA, y, A)
            list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X, y, lamda, etaj, n, p, threshold)

            # Tính p-value
            p_value = util.p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov)

            # Tính TPR hoặc FPR tùy thuộc vào mode
            if mode == 'TPR':
                if beta_vec[j_selected] != 0:  # Đặc trưng quan trọng
                    correctly_detected += 1
                    if p_value < 0.05:  # Bác bỏ null hypothesis
                        correctly_rejected += 1
            else:  # FPR mode
                if beta_vec[j_selected] == 0 and p_value < 0.05:  # Đặc trưng không quan trọng nhưng bị chọn nhầm
                    correctly_detected += 1

        # Tính TPR hoặc FPR cho mỗi trial
        if correctly_detected > 0:
            TPR = correctly_rejected / correctly_detected if mode == 'TPR' else correctly_detected / total_relevant_features
        else:
            TPR = 0

        TPR_trials.append(TPR)

    # Tính trung bình TPR hoặc FPR sau các trials
    TPR_mean = np.mean(TPR_trials)

    return TPR_mean


def run_experiment():
    n_values = [100, 200, 300, 400, 500]  # Kích thước mẫu khác nhau
    p = 5
    lamda = 1
    num_trials = 100

    # Chạy TPR với beta_vec chứa hai đặc trưng quan trọng
    beta_vec_tpr = [2, 2, 0, 0, 0]  # Hai đặc trưng quan trọng
    TPR_results = []
    lamda = 0.05
    beta_vec = [1, 1, 0, 0, 0]
    for n in n_values:
        print(f"Running TPR experiments for n = {n}")
        TPR = run_tpr_fpr(n, p, lamda, beta_vec_tpr, num_trials, mode='TPR')
        TPR_results.append(TPR)
        print(f"n = {n}: Mean TPR = {TPR}")

    # Chạy FPR với tất cả các beta_vec bằng 0
    beta_vec_fpr = [0, 0, 0, 0, 0]  # Tất cả đặc trưng không quan trọng
    FPR_results = []

    for n in n_values:
        print(f"Running FPR experiments for n = {n}")
        FPR = run_tpr_fpr(n, p, lamda, beta_vec_fpr, num_trials, mode='FPR')
        FPR_results.append(FPR)
        print(f"n = {n}: Mean FPR = {FPR}")

    # Lưu kết quả và vẽ biểu đồ
    plot_results(n_values, TPR_results, FPR_results)


def plot_results(n_values, TPR_results, FPR_results):
    plt.figure(figsize=(10, 5))

    # Vẽ FPR
    plt.subplot(1, 2, 1)
    plt.plot(n_values, FPR_results, 'o-', label='FPR')
    plt.xlabel('Sample size')
    plt.ylabel('FPR')
    plt.ylim(0, 1)  # Đảm bảo FPR nằm trong khoảng từ 0 đến 1
    plt.title('False Positive Rate')
    plt.legend()

    # Vẽ TPR
    plt.subplot(1, 2, 2)
    plt.plot(n_values, TPR_results, 'o-', label='TPR', color='green')
    plt.xlabel('Sample size')
    plt.ylabel('TPR')
    plt.ylim(0, 1)  # Đảm bảo TPR nằm trong khoảng từ 0 đến 1
    plt.title('True Positive Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('tpr_fpr_results.png', dpi=300)  # Lưu biểu đồ
    plt.show()


if __name__ == '__main__':
    run_experiment()
