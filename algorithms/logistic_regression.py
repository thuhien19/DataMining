from sklearn.linear_model import LogisticRegression


# ==================================================
# LOGISTIC REGRESSION
# ==================================================

def run_logistic_regression(
    X,
    y
):

    # Khởi tạo mô hình
    model = LogisticRegression(
        max_iter=1000
    )

    # Huấn luyện
    model.fit(
        X,
        y
    )

    return model