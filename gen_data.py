import numpy as np


def generate(n, p, beta_vec):
    X = []  # Khởi tạo danh sách để lưu các mẫu đầu vào X
    y = []  # Khởi tạo danh sách để lưu các giá trị đầu ra bị nhiễu
    true_y = []  # Khởi tạo danh sách để lưu các giá trị đầu ra "thực" (không bị nhiễu)

    # Vòng lặp tạo n mẫu
    for i in range(n):
        X.append([])  # Khởi tạo danh sách con cho mỗi hàng của X
        yi = 0  # Giá trị đầu ra khởi đầu (sẽ được tính dần)
        
        # Vòng lặp qua từng đặc trưng p
        for j in range(p):
            xij = np.random.normal(0, 1)  # Sinh giá trị xij từ phân phối chuẩn N(0, 1)
            X[i].append(xij)  # Thêm xij vào hàng hiện tại của X
            yi = yi + xij * beta_vec[j]  # Cộng dồn đóng góp của xij vào giá trị đầu ra

        true_y.append(yi)  # Lưu giá trị "thực" của yi trước khi thêm nhiễu
        noise = np.random.normal(0, 1)  # Sinh nhiễu từ phân phối chuẩn N(0, 1)
        yi = yi + noise  # Thêm nhiễu vào giá trị đầu ra
        y.append(yi)  # Lưu giá trị đầu ra bị nhiễu vào danh sách y

    # Chuyển đổi X, y, true_y từ danh sách sang mảng numpy
    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y  # Trả về các mảng X, y, true_y


def generate_non_normal(n, p, beta_vec):
    X = []  # Khởi tạo danh sách để lưu các mẫu đầu vào X
    y = []  # Khởi tạo danh sách để lưu các giá trị đầu ra bị nhiễu
    true_y = []  # Khởi tạo danh sách để lưu các giá trị đầu ra "thực" (không bị nhiễu)

    # Vòng lặp tạo n mẫu
    for i in range(n):
        X.append([])  # Khởi tạo danh sách con cho mỗi hàng của X
        yi = 0  # Giá trị đầu ra khởi đầu (sẽ được tính dần)

        # Vòng lặp qua từng đặc trưng p
        for j in range(p):
            xij = np.random.normal(0, 1)  # Sinh giá trị xij từ phân phối chuẩn N(0, 1)
            X[i].append(xij)  # Thêm xij vào hàng hiện tại của X
            yi = yi + xij * beta_vec[j]  # Cộng dồn đóng góp của xij vào giá trị đầu ra

        true_y.append(yi)  # Lưu giá trị "thực" của yi trước khi thêm nhiễu

        noise = np.random.normal(0, 1)  # Sinh nhiễu từ phân phối chuẩn N(0, 1)
        # Các dòng dưới là những phân phối nhiễu thay thế khác (bình luận lại để không được dùng):
        # noise = np.random.laplace(0, 1)  # Sinh nhiễu từ phân phối Laplace
        # noise = skewnorm.rvs(a=10, loc=0, scale=1)  # Sinh nhiễu từ phân phối Skew-Normal
        # noise = np.random.standard_t(20)  # Sinh nhiễu từ phân phối t với 20 bậc tự do

        yi = yi + noise  # Thêm nhiễu vào giá trị đầu ra
        y.append(yi)  # Lưu giá trị đầu ra bị nhiễu vào danh sách y

    # Chuyển đổi X, y, true_y từ danh sách sang mảng numpy
    X = np.array(X)
    y = np.array(y)
    true_y = np.array(true_y)

    return X, y, true_y  # Trả về các mảng X, y, true_y
