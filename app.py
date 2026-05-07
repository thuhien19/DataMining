import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate_model
from utils.visualization import plot_confusion_matrix

from algorithms.naive_bayes import run_naive_bayes
from algorithms.decision_tree import run_decision_tree
from algorithms.logistic_regression import run_logistic_regression

from algorithms.random_forest import run_random_forest


st.set_page_config(
    page_title="Data Mining App",
    layout="wide"
)

st.title("HỆ THỐNG CHẠY THUẬT TOÁN KHAI PHÁ DỮ LIỆU")

st.sidebar.header("MENU")

algorithm = st.sidebar.selectbox(
    "Chọn thuật toán",
    [
        "Naive Bayes",
        "Decision Tree",
        "Logistic Regression",
        "Random Forest"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )

    st.subheader("Dữ liệu ban đầu")
    st.dataframe(df)

    st.info("Hệ thống mặc định cột cuối cùng là thuộc tính quyết định.")

    X_train, X_test, y_train, y_test, target_col, encoders = preprocess_data(df)

    st.write("Thuộc tính quyết định:", target_col)

    st.write("Số dòng train:", len(X_train))
    st.write("Số dòng test:", len(X_test))

    if st.button("Chạy thuật toán"):

        if algorithm == "Naive Bayes":
            model = run_naive_bayes(X_train, y_train)

        elif algorithm == "Decision Tree":
            model = run_decision_tree(X_train, y_train)

        elif algorithm == "Logistic Regression":
            model = run_logistic_regression(X_train, y_train)

        elif algorithm == "SVM":
            model = run_svm(X_train, y_train)

        elif algorithm == "Random Forest":
            model = run_random_forest(X_train, y_train)

        accuracy, report, matrix, y_pred = evaluate_model(
            model,
            X_test,
            y_test
        )

        st.success("Huấn luyện mô hình thành công!")

        st.subheader("Kết quả dự đoán")
        result_df = pd.DataFrame({
            "Thực tế": y_test.values,
            "Dự đoán": y_pred
        })

        st.dataframe(result_df)

        st.subheader("Độ chính xác Accuracy")
        st.metric("Accuracy", round(accuracy, 4))

        st.subheader("Classification Report")
        st.text(report)

        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(matrix)
        st.pyplot(fig)

else:
    st.warning("Vui lòng upload file CSV để bắt đầu.")