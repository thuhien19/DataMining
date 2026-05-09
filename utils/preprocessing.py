from sklearn.preprocessing import LabelEncoder
import pandas as pd

def basic_preprocessing(df):

    data = df.copy()

    # ==========================================
    # XÓA CỘT ID / STT
    # ==========================================

    ignore_cols = [
        "STT",
        "Id",
        "ID",
        "#",
        "Transaction ID"
    ]

    data = data.drop(
        columns=[
            col
            for col in ignore_cols
            if col in data.columns
        ],
        errors="ignore"
    )

    # ==========================================
    # XỬ LÝ DỮ LIỆU THIẾU
    # ==========================================

    for col in data.columns:

        # dữ liệu số
        if pd.api.types.is_numeric_dtype(
            data[col]
        ):

            data[col] = data[col].fillna(
                data[col].mean()
            )

        # dữ liệu chuỗi
        else:

            mode_value = data[col].mode()

            if not mode_value.empty:

                data[col] = data[col].fillna(
                    mode_value[0]
                )

    return data

def preprocess_classification_data(df):
    data = df.copy()

    if "STT" in data.columns:
        data = data.drop(columns=["STT"])

    target_col = data.columns[-1]
    feature_cols = data.columns[:-1].tolist()

    encoders = {}

    encoded_data = data.copy()

    for col in encoded_data.columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        encoders[col] = le

    X = encoded_data[feature_cols]
    y = encoded_data[target_col]

    return X, y, target_col, feature_cols, encoders, data, encoded_data