import numpy as np
from sklearn import linear_model
import util  # Giả định bạn có một module util chứa các hàm cần thiết như construct_A_XA_Ac_XAc_bhA và compute_yz

# Hàm tính thương số giữa tử số và mẫu số, trả về vô cực nếu mẫu số bằng 0 hoặc nếu thương số <= 0
def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.Inf  # Trả về vô cực nếu mẫu số bằng 0

    quotient = numerator / denominator

    if quotient <= 0:
        return np.Inf  # Nếu thương số <= 0, trả về vô cực

    return quotient

# Hàm chạy Lasso với dữ liệu mới (yz) và tìm các chỉ số active set, cùng với hệ số hồi quy, etaj và các thông số khác
def parametric_lasso_cv(X, yz, lamda, b, n, p):
    yz_flatten = yz.flatten()  # Làm phẳng vector yz để sử dụng trong hàm Lasso

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)  # Tạo mô hình Lasso
    clf.fit(X, yz_flatten)  # Huấn luyện mô hình Lasso trên dữ liệu yz
    bhz = clf.coef_  # Lấy hệ số hồi quy từ mô hình

    # Phân loại các biến thành active (A) và inactive (Ac), cùng với ma trận thiết kế tương ứng
    Az, XAz, Acz, XAcz, bhAz = util.construct_A_XA_Ac_XAc_bhA(X, bhz, n, p)

    etaAz = np.array([])  # Khởi tạo etaAz

    # Nếu tồn tại ma trận thiết kế cho các biến active, tính etaAz
    if XAz is not None:
        inv = np.linalg.pinv(np.dot(XAz.T, XAz))  # Tính ma trận pseudo-inverse
        invXAzT = np.dot(inv, XAz.T)  # Tính inv(XAz^T XAz) XAz^T
        etaAz = np.dot(invXAzT, b)  # Tính etaAz

    # Khởi tạo các vector rỗng cho shAz và gammaAz
    shAz = np.array([])
    gammaAz = np.array([])

    # Nếu tồn tại ma trận thiết kế cho các biến inactive, tính shAz và gammaAz
    if XAcz is not None:
        if XAz is None:
            e1 = yz
        else:
            e1 = yz - np.dot(XAz, bhAz)  # Phần dư (residual)
        
        e2 = np.dot(XAcz.T, e1)  # Chiếu phần dư lên các biến inactive
        shAz = e2 / (lamda * n)  # Tính shAz

        # Tính gammaAz dựa trên b
        if XAz is None:
            gammaAz = (np.dot(XAcz.T, b)) / n
        else:
            gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), etaAz)) / n

    # Làm phẳng các vector để xử lý dễ dàng hơn
    bhAz = bhAz.flatten()
    etaAz = etaAz.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()

    # Khởi tạo giá trị min1 và min2 là vô cực
    min1 = np.Inf
    min2 = np.Inf

    # Tính min1: Tìm thương số nhỏ nhất giữa -bhAz[j] và etaAz[j]
    for j in range(len(etaAz)):
        numerator = - bhAz[j]
        denominator = etaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    # Tính min2: Tìm thương số nhỏ nhất giữa (sign(gammaAz[j]) - shAz[j]) * lamda và gammaAz[j]
    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j]) * lamda
        denominator = gammaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min2:
            min2 = quotient

    # Trả về giá trị nhỏ hơn giữa min1 và min2, cùng với các biến liên quan
    return min(min1, min2), Az, bhz, etaAz, bhAz

# Hàm chạy quy trình Lasso với kiểm định giá trị z trong khoảng từ -threshold đến threshold
def run_parametric_lasso_cv(X, lamda, n, p, threshold, a, b):

    zk = -threshold  # Khởi tạo zk là giá trị âm của threshold

    # Khởi tạo các danh sách để lưu kết quả
    list_zk = [zk]
    list_active_set = []
    list_bhz = []
    list_etaAkz = []
    list_bhAz = []

    # Lặp qua zk cho đến khi zk đạt đến threshold
    while zk < threshold:
        yz = a + b * zk  # Tính yz với zk hiện tại
        skz, Akz, bhkz, etaAkz, bhAkz = parametric_lasso_cv(X, yz, lamda, b, n, p)  # Chạy Lasso trên yz

        zk = zk + skz + 0.0001  # Cập nhật zk

        if zk < threshold:
            list_zk.append(zk)  # Thêm zk vào danh sách nếu chưa vượt quá threshold
        else:
            list_zk.append(threshold)  # Nếu vượt quá, thêm threshold vào danh sách

        # Lưu các giá trị tương ứng vào danh sách
        list_active_set.append(Akz)
        list_bhz.append(bhkz)
        list_etaAkz.append(etaAkz)
        list_bhAz.append(bhAkz)

    # Trả về các danh sách kết quả
    return list_zk, list_bhz, list_active_set, list_etaAkz, list_bhAz

# Hàm chạy Lasso với một giá trị zk cụ thể
def parametric_lasso(X, yz, lamda, b, n, p):
    yz_flatten = yz.flatten()  # Làm phẳng vector yz

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)  # Tạo mô hình Lasso
    clf.fit(X, yz_flatten)  # Huấn luyện mô hình
    bhz = clf.coef_  # Lấy hệ số hồi quy từ mô hình

    # Phân loại các biến active và inactive
    Az, XAz, Acz, XAcz, bhAz = util.construct_A_XA_Ac_XAc_bhA(X, bhz, n, p)

    etaAz = np.array([])

    # Nếu tồn tại ma trận thiết kế cho các biến active, tính etaAz
    if XAz is not None:
        inv = np.linalg.pinv(np.dot(XAz.T, XAz))
        invXAzT = np.dot(inv, XAz.T)
        etaAz = np.dot(invXAzT, b)

    shAz = np.array([])
    gammaAz = np.array([])

    # Tính shAz và gammaAz nếu có ma trận thiết kế cho các biến inactive
    if XAcz is not None:
        if XAz is None:
            e1 = yz
        else:
            e1 = yz - np.dot(XAz, bhAz)

        e2 = np.dot(XAcz.T, e1)
        shAz = e2 / (lamda * n)

        if XAz is None:
            gammaAz = (np.dot(XAcz.T, b)) / n
        else:
            gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), etaAz)) / n

    # Làm phẳng các vector để xử lý dễ dàng hơn
    bhAz = bhAz.flatten()
    etaAz = etaAz.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()

    min1 = np.Inf
    min2 = np.Inf

    # Tính min1: Tìm thương số nhỏ nhất giữa -bhAz[j] và etaAz[j]
    for j in range(len(etaAz)):
        numerator = - bhAz[j]
        denominator = etaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    # Tính min2: Tìm thương số nhỏ nhất giữa (sign(gammaAz[j]) - shAz[j]) * lamda và gammaAz[j]
    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j]) * lamda
        denominator = gammaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min2:
            min2 = quotient

    # Trả về giá trị nhỏ nhất giữa min1 và min2, cùng với các biến active và hệ số hồi quy
    return min(min1, min2), Az, bhz

# Hàm chạy Lasso cho một tập các giá trị zk, sử dụng threshold để xác định phạm vi zk
def run_parametric_lasso(X, y, lamda, etaj, n, p, threshold):

    zk = -threshold  # Khởi tạo zk là âm của threshold

    # Khởi tạo các danh sách để lưu kết quả
    list_zk = [zk]
    list_active_set = []
    list_bhz = []

    # Lặp qua zk cho đến khi zk đạt đến threshold
    while zk < threshold:
        yz, b = util.compute_yz(y, etaj, zk, n)  # Tính yz và b với zk hiện tại

        skz, Akz, bhkz = parametric_lasso(X, yz, lamda, b, n, p)  # Chạy Lasso trên yz

        zk = zk + skz + 0.0001  # Cập nhật zk

        if zk < threshold:
            list_zk.append(zk)  # Thêm zk vào danh sách nếu chưa vượt quá threshold
        else:
            list_zk.append(threshold)  # Nếu vượt quá, thêm threshold vào danh sách

        # Lưu các giá trị tương ứng vào danh sách
        list_active_set.append(Akz)
        list_bhz.append(bhkz)

    # Trả về các danh sách kết quả
    return list_zk, list_bhz, list_active_set
