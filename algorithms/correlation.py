import pandas as pd


# ==========================================
# HÀM TÍNH TRUNG BÌNH
# ==========================================

def mean(data):

    return sum(data) / len(data)


# ==========================================
# HÀM TÍNH PEARSON CORRELATION
# ==========================================

def pearson_correlation(x, y):

    if len(x) != len(y):

        raise ValueError(
            "Hai danh sách phải có cùng độ dài"
        )

    n = len(x)

    mean_x = mean(x)
    mean_y = mean(y)

    # ==========================================
    # TỬ SỐ
    # ==========================================

    numerator = sum(
        (x[i] - mean_x) * (y[i] - mean_y)
        for i in range(n)
    )

    # ==========================================
    # MẪU SỐ
    # ==========================================

    denominator_x = sum(
        (x[i] - mean_x) ** 2
        for i in range(n)
    )

    denominator_y = sum(
        (y[i] - mean_y) ** 2
        for i in range(n)
    )

    denominator = (
        denominator_x * denominator_y
    ) ** 0.5

    if denominator == 0:

        return 0

    return numerator / denominator


# ==========================================
# HÀM CHẠY TƯƠNG QUAN
# ==========================================

def run_correlation(df):

    # Copy dữ liệu
    df_encoded = df.copy()

    # ==========================================
    # MÃ HÓA TOÀN BỘ CỘT
    # ==========================================

    for col in df_encoded.columns:

        # Nếu KHÔNG phải số
        if not pd.api.types.is_numeric_dtype(
            df_encoded[col]
        ):

            df_encoded[col] = pd.factorize(
                df_encoded[col].astype(str)
            )[0]

    # ==========================================
    # TẠO MA TRẬN TƯƠNG QUAN
    # ==========================================

    columns = df_encoded.columns

    correlation_matrix = pd.DataFrame(
        index=columns,
        columns=columns
    )

    # ==========================================
    # TÍNH TƯƠNG QUAN
    # ==========================================

    for col1 in columns:

        for col2 in columns:

            x = df_encoded[col1].tolist()
            y = df_encoded[col2].tolist()

            r = pearson_correlation(x, y)

            correlation_matrix.loc[
                col1,
                col2
            ] = round(r, 3)

    return (
        df_encoded,
        correlation_matrix
    )