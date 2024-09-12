import numpy as np
from mpmath import mp

# Đặt độ chính xác của mpmath cho tính toán p-value
mp.dps = 500

# Hàm xây dựng vector dấu s từ các hệ số bh của mô hình
def construct_s(bh):
    """
    Tạo vector dấu từ các hệ số bh.
    Parameters:
    - bh: các hệ số của mô hình.
    
    Returns:
    - s: vector dấu tương ứng với các hệ số bh.
    """
    s = []
    for bhj in bh:
        if bhj != 0:
            s.append(np.sign(bhj))  # Thêm dấu của hệ số vào s

    s = np.array(s).reshape((len(s), 1))  # Định dạng lại s thành vector cột
    return s

# Hàm phân loại các biến vào nhóm A (active) và Ac (inactive)
def construct_A_XA_Ac_XAc_bhA(X, bh, n, p):
    """
    Phân chia các biến vào nhóm active A và inactive Ac, đồng thời lấy dữ liệu thiết kế tương ứng.
    Parameters:
    - X: ma trận thiết kế.
    - bh: hệ số hồi quy.
    - n: số lượng mẫu.
    - p: số lượng biến.
    
    Returns:
    - A: danh sách các chỉ số biến active.
    - XA: ma trận thiết kế tương ứng với biến active.
    - Ac: danh sách các chỉ số biến inactive.
    - XAc: ma trận thiết kế tương ứng với biến inactive.
    - bhA: hệ số hồi quy tương ứng với biến active.
    """
    A = []   # Danh sách biến active
    Ac = []  # Danh sách biến inactive
    bhA = [] # Hệ số của các biến active

    # Phân loại các biến vào nhóm A và Ac
    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)       # Nếu hệ số khác 0, biến thuộc nhóm active
            bhA.append(bhj)   # Lưu lại hệ số tương ứng
        else:
            Ac.append(j)      # Nếu hệ số bằng 0, biến thuộc nhóm inactive

    XA = X[:, A]   # Ma trận thiết kế cho các biến active
    XAc = X[:, Ac] # Ma trận thiết kế cho các biến inactive
    bhA = np.array(bhA).reshape((len(A), 1))  # Định dạng lại bhA thành vector cột

    return A, XA, Ac, XAc, bhA

# Hàm kiểm tra điều kiện KKT (Karush-Kuhn-Tucker)
def check_KKT(XA, XAc, y, bhA, lamda, n):
    """
    Kiểm tra điều kiện KKT cho mô hình hồi quy.
    Parameters:
    - XA: ma trận thiết kế cho biến active.
    - XAc: ma trận thiết kế cho biến inactive.
    - y: vector mục tiêu.
    - bhA: hệ số của biến active.
    - lamda: hệ số regularization lambda.
    - n: số lượng mẫu.
    """
    print("\nCheck Active")
    e1 = y - np.dot(XA, bhA)  # Phần dư
    e2 = np.dot(XA.T, e1)     # Phần điều kiện active
    print(e2 / (lamda * n))    # In kết quả kiểm tra cho nhóm active

    if XAc is not None:
        print("\nCheck In Active")
        e1 = y - np.dot(XA, bhA)  # Phần dư
        e2 = np.dot(XAc.T, e1)    # Phần điều kiện inactive
        print(e2 / (lamda * n))   # In kết quả kiểm tra cho nhóm inactive

# Hàm tính thống kê kiểm định cho biến j
def construct_test_statistic(j, XA, y, A):
    """
    Xây dựng thống kê kiểm định cho biến j.
    Parameters:
    - j: chỉ số của biến đang kiểm định.
    - XA: ma trận thiết kế cho các biến active.
    - y: vector mục tiêu.
    - A: danh sách các biến active.
    
    Returns:
    - etaj: vector chiếu.
    - etajTy: phép chiếu của y lên etaj.
    """
    # Xây dựng vector chỉ số ej (1 cho j, 0 cho các chỉ số khác)
    ej = []
    for each_j in A:
        if j == each_j:
            ej.append(1)
        else:
            ej.append(0)
    ej = np.array(ej).reshape((len(A), 1))

    # Tính ma trận pseudo-inverse của XA
    inv = np.linalg.pinv(np.dot(XA.T, XA))
    XAinv = np.dot(XA, inv)  #XAinv là X_plus

    # Tính etaj
    etaj = np.dot(XAinv, ej)

    # Tính etaj^T y
    etajTy = np.dot(etaj.T, y)[0][0]

    return etaj, etajTy

# Hàm tính toán vector yz và b dựa trên etaj và zk
def compute_yz(y, etaj, zk, n):
    """
    Tính toán vector yz và b dựa trên etaj và zk.
    Parameters:
    - y: vector mục tiêu.
    - etaj: vector chiếu.
    - zk: giá trị z hiện tại.
    - n: số lượng mẫu.
    
    Returns:
    - yz: vector điều chỉnh.
    - b: hệ số chiếu.
    """
    sq_norm = (np.linalg.norm(etaj))**2  # Bình phương chuẩn của etaj

    e1 = np.identity(n) - (np.dot(etaj, etaj.T)) / sq_norm  # Phần chiếu trực giao
    a = np.dot(e1, y)  # Điều chỉnh y

    b = etaj / sq_norm  # Hệ số chiếu

    yz = a + b * zk  # Tạo ra vector điều chỉnh yz

    return yz, b

# Hàm tính pivot dựa trên khoảng z cho từng khoảng active set
def pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_mu, type):
    """
    Tính pivot dựa trên khoảng z.
    Parameters:
    - A: chỉ số biến active.
    - bh: hệ số hồi quy.
    - list_active_set: danh sách các active set.
    - list_zk: danh sách giá trị z.
    - list_bhz: danh sách hệ số hồi quy bh tương ứng.
    - etaj: vector chiếu.
    - etajTy: phép chiếu của y lên etaj.
    - cov: ma trận hiệp phương sai.
    - tn_mu: trung bình của phân phối truncated normal.
    - type: loại kiểm định ('A' hoặc 'As').
    
    Returns:
    - Pivot p-value dựa trên khoảng z.
    """
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]  # Độ lệch chuẩn

    # Tính khoảng z dựa trên loại kiểm định
    z_interval = []
    for i in range(len(list_active_set)):
        if type == 'As':
            if np.array_equal(np.sign(bh), np.sign(list_bhz[i])):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

        if type == 'A':
            if np.array_equal(A, list_active_set[i]):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    # Tinh chỉnh khoảng z
    new_z_interval = []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)
    z_interval = new_z_interval

    numerator = 0
    denominator = 0

    # Tính toán giá trị numerator và denominator cho pivot
    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]
        denominator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)
        if etajTy >= ar:
            numerator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator += mp.ncdf((etajTy - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)

    # Trả về pivot p-value
    if denominator != 0:
        return float(numerator / denominator)
    else:
        return None

# Hàm tính pivot với khoảng z cho trước
def pivot_with_specified_interval(z_interval, etaj, etajTy, cov, tn_mu):
    """
    Tính pivot dựa trên khoảng z được chỉ định.
    Parameters:
    - z_interval: khoảng z.
    - etaj: vector chiếu.
    - etajTy: phép chiếu của y lên etaj.
    - cov: ma trận hiệp phương sai.
    - tn_mu: trung bình của phân phối truncated normal.
    
    Returns:
    - Pivot p-value dựa trên khoảng z.
    """
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    numerator = 0
    denominator = 0

    # Tính numerator và denominator cho pivot với khoảng z cho trước
    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]
        denominator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)
        if etajTy >= ar:
            numerator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator += mp.ncdf((etajTy - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)

    if denominator != 0:
        return float(numerator / denominator)
    else:
        return None

# Hàm tính p-value
def p_value(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov):
    """
    Tính toán p-value từ pivot.
    Parameters:
    - A: chỉ số biến active.
    - bh: hệ số hồi quy.
    - list_active_set: danh sách các active set.
    - list_zk: danh sách giá trị z.
    - list_bhz: danh sách hệ số hồi quy bh tương ứng.
    - etaj: vector chiếu.
    - etajTy: phép chiếu của y lên etaj.
    - cov: ma trận hiệp phương sai.
    
    Returns:
    - p-value dựa trên kiểm định pivot.
    """
    value = pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, 0, 'A')
    return 2 * min(1 - value, value)  # Tính p-value 2-tail
