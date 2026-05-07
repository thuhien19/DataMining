from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    data = df.copy()

    # Bỏ cột STT nếu có
    if "STT" in data.columns:
        data = data.drop(columns=["STT"])

    # Cột cuối cùng là thuộc tính quyết định
    target_col = data.columns[-1]

    # Mã hóa dữ liệu dạng chữ
    encoders = {}

    for col in data.columns:
        if data[col].dtype == "object":
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))
            encoders[col] = encoder

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, target_col, encoders