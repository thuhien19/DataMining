import streamlit as st
import pandas as pd

from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate_model
from utils.visualization import plot_confusion_matrix

from algorithms.correlation import run_correlation
from algorithms.apriori import run_apriori
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
        "Correlation",
        "Apriori",
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


    # ==================================================
    # APRIORI
    # ==================================================

    if algorithm == "Apriori":

        min_support = st.sidebar.slider(
            "Min Support",
            0.1,
            1.0,
            0.3,
            0.1
        )

        min_confidence = st.sidebar.slider(
            "Min Confidence",
            0.1,
            1.0,
            0.6,
            0.1
        )

        if st.button(
            "Chạy Apriori",
            key="apriori_btn"
        ):

            frequent_itemsets, rules, encoded_df = run_apriori(
                df,
                min_support,
                min_confidence
            )

            st.success("Chạy thuật toán Apriori thành công!")

            # ==========================================
            # DỮ LIỆU ENCODE
            # ==========================================

            st.subheader("DỮ LIỆU SAU KHI ENCODE")

            st.dataframe(encoded_df)

            # ==========================================
            # TẬP PHỔ BIẾN
            # ==========================================

            st.subheader(
                f"TẬP PHỔ BIẾN THỎA min_support = {min_support}"
            )

            st.dataframe(frequent_itemsets)

            # ==========================================
            # LUẬT KẾT HỢP
            # ==========================================

            st.subheader(
                f"LUẬT KẾT HỢP THỎA min_confidence = {min_confidence}"
            )

            if not rules.empty:

                result = rules[
                    [
                        "antecedents",
                        "consequents",
                        "support",
                        "confidence",
                        "lift"
                    ]
                ]

                # Format đẹp hơn
                result["antecedents"] = result[
                    "antecedents"
                ].apply(
                    lambda x: ", ".join(list(x))
                )

                result["consequents"] = result[
                    "consequents"
                ].apply(
                    lambda x: ", ".join(list(x))
                )

                st.dataframe(result)

            else:

                st.warning(
                    "Không tìm thấy luật kết hợp phù hợp."
                )
    # ==================================================
    # CORRELATION
    # ==================================================

    elif algorithm == "Correlation":

        if st.button(
            "Tính tương quan",
            key="correlation_btn"
        ):

            encoded_df, correlation_matrix = run_correlation(df)

            st.success(
                "Tính ma trận tương quan thành công!"
            )
            
            # ==========================================
            # MA TRẬN TƯƠNG QUAN
            # ==========================================

            st.subheader(
                "MA TRẬN TƯƠNG QUAN"
            )

            st.dataframe(correlation_matrix)

            # ==========================================
            # HEATMAP
            # ==========================================

            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            sns.heatmap(
                correlation_matrix.astype(float),
                annot=True,
                cmap="coolwarm",
                ax=ax
            )

            st.subheader(
                "HEATMAP TƯƠNG QUAN"
            )

            st.pyplot(fig)
    # ==================================================
    # CLASSIFICATION
    # ==================================================

    else:

        st.info(
            "Hệ thống mặc định cột cuối cùng là thuộc tính quyết định."
        )

        X_train, X_test, y_train, y_test, target_col, encoders = preprocess_data(df)

        st.write("Thuộc tính quyết định:", target_col)

        st.write("Số dòng train:", len(X_train))
        st.write("Số dòng test:", len(X_test))

        if st.button(
            "Chạy Classification",
            key="classification_btn"
        ):

            if algorithm == "Naive Bayes":

                model = run_naive_bayes(
                    X_train,
                    y_train
                )

            elif algorithm == "Decision Tree":

                model = run_decision_tree(
                    X_train,
                    y_train
                )

            elif algorithm == "Logistic Regression":

                model = run_logistic_regression(
                    X_train,
                    y_train
                )

            elif algorithm == "Random Forest":

                model = run_random_forest(
                    X_train,
                    y_train
                )

            accuracy, report, matrix, y_pred = evaluate_model(
                model,
                X_test,
                y_test
            )

            st.success(
                "Huấn luyện mô hình thành công!"
            )

            # ==========================================
            # KẾT QUẢ DỰ ĐOÁN
            # ==========================================

            st.subheader("KẾT QUẢ DỰ ĐOÁN")

            result_df = pd.DataFrame({
                "Thực tế": y_test.values,
                "Dự đoán": y_pred
            })

            st.dataframe(result_df)

            # ==========================================
            # ACCURACY
            # ==========================================

            st.subheader("ĐỘ CHÍNH XÁC ACCURACY")

            st.metric(
                "Accuracy",
                round(accuracy, 4)
            )

            # ==========================================
            # REPORT
            # ==========================================

            st.subheader("CLASSIFICATION REPORT")

            st.text(report)

            # ==========================================
            # CONFUSION MATRIX
            # ==========================================

            st.subheader("CONFUSION MATRIX")

            fig = plot_confusion_matrix(matrix)

            st.pyplot(fig)

else:

    st.warning(
        "Vui lòng upload file CSV để bắt đầu."
    )