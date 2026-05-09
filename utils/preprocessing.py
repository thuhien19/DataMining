import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)
from sklearn.model_selection import (
    train_test_split
)

def basic_preprocessing(df):

    data = df.copy()

    # XÓA CỘT ID / STT
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

    # XỬ LÝ DỮ LIỆU THIẾU

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

# ENCODE DATA
def encode_data(data):

    encoded_data = data.copy()

    encoders = {}

    for col in encoded_data.columns:

        # Encode dữ liệu chuỗi
        if not pd.api.types.is_numeric_dtype(
            encoded_data[col]
        ):

            le = LabelEncoder()

            encoded_data[col] = le.fit_transform(
                encoded_data[col].astype(str)
            )

            encoders[col] = le

    return encoded_data, encoders

def preprocess_classification_data(df):
    data = df.copy()

    data = basic_preprocessing(df)


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


# PREPROCESS CHO LOGISTIC / RAMDOM
def preprocess_scaled_data(df):

    data = basic_preprocessing(df)

    target_col = data.columns[-1]

    feature_cols = data.columns[:-1].tolist()

    encoded_data, encoders = encode_data(data)

    X = encoded_data[feature_cols]

    y = encoded_data[target_col]

    # SCALE
    scaler = MinMaxScaler()

    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        target_col,
        feature_cols,
        encoders,
        data,
        encoded_data
    )