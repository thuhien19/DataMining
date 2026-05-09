from sklearn.preprocessing import LabelEncoder

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