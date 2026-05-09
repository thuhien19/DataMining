import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler
)

from utils.preprocessing import (
    basic_preprocessing
)


def predict_new_data(
    predict_df,
    model,
    feature_cols,
    encoders,
    target_col,
    encoded_data
):

    # Preprocess chung
    predict_data = basic_preprocessing(
        predict_df
    )

    # Encode categorical
    for col in predict_data.columns:

        if col in encoders:

            predict_data[col] = encoders[
                col
            ].transform(
                predict_data[col].astype(str)
            )

    # Chỉ lấy feature
    predict_data = predict_data[
        feature_cols
    ]

    # Scale
    scaler = MinMaxScaler()

    scaler.fit(
        encoded_data[feature_cols]
    )

    predict_data = pd.DataFrame(
        scaler.transform(predict_data),
        columns=feature_cols
    )

    # Predict
    predictions = model.predict(
        predict_data
    )

    predicted_labels = encoders[
        target_col
    ].inverse_transform(
        predictions
    )

    # Result
    result_df = predict_df.copy()

    result_df[
        "Prediction"
    ] = predicted_labels

    return result_df