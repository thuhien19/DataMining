import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from utils.evaluation import evaluate_model
from utils.visualization import plot_confusion_matrix
from utils.preprocessing import preprocess_classification_data

from algorithms.naive_bayes import naive_bayes_predict
from algorithms.correlation import run_correlation
from algorithms.apriori import run_apriori
from algorithms.decision_tree import (
    build_tree,
    draw_categorical_tree
)
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

        # ==================================================
    # CLASSIFICATION
    # ==================================================

    else:

        st.info(
            "Hệ thống mặc định cột cuối cùng là thuộc tính quyết định."
        )

        X, y, target_col, feature_cols, encoders, original_data, encoded_data = preprocess_classification_data(df)

        st.write("Thuộc tính quyết định:", target_col)

        st.subheader("Các thuộc tính điều kiện")
        st.write(feature_cols)

        # ==================================================
        # NAIVE BAYES
        # ==================================================

        if algorithm == "Naive Bayes":

            st.subheader("DỰ ĐOÁN BẰNG NAIVE BAYES")

            nb_type = st.radio(
                "Chọn loại Naive Bayes",
                [
                    "Naive Bayes thường",
                    "Naive Bayes có làm trơn Laplace"
                ]
            )

            input_values = {}

            st.write("Chọn giá trị cho từng thuộc tính:")

            for col in feature_cols:
                values = original_data[col].unique().tolist()

                input_values[col] = st.selectbox(
                    f"{col}",
                    values,
                    key=f"nb_{col}"
                )

            if st.button(
                "Dự đoán Naive Bayes",
                key="nb_predict_btn"
            ):

                laplace = True if nb_type == "Naive Bayes có làm trơn Laplace" else False

                predicted_class, result_df = naive_bayes_predict(
                    original_data,
                    input_values,
                    target_col,
                    laplace=laplace
                )

                st.success("Dự đoán thành công!")

                st.subheader("Dữ liệu cần dự đoán")

                input_df = pd.DataFrame([input_values])
                st.dataframe(input_df)

                st.subheader("Bảng xác suất")

                st.dataframe(result_df)

                st.subheader("Kết quả dự đoán")

                st.metric(
                    "Lớp dự đoán",
                    predicted_class
                )

        # ==================================================
        # DECISION TREE
        # ==================================================

        elif algorithm == "Decision Tree":

            st.subheader("CÂY QUYẾT ĐỊNH")

            tree_type = st.radio(
                "Chọn tiêu chí phân chia",
                [
                    "Information Gain / Entropy",
                    "Gini Index"
                ]
            )

            if tree_type == "Information Gain / Entropy":
                criterion = "entropy"
                title = "Cây quyết định ID3 - Information Gain"
            else:
                criterion = "gini"
                title = "Cây quyết định CART - Gini Index"

            if st.button(
                "Xây dựng cây quyết định",
                key="dt_btn"
            ):

                tree = build_tree(
                    original_data,
                    feature_cols,
                    target_col,
                    criterion=criterion
                )

                st.success("Xây dựng cây quyết định thành công!")

                st.subheader("Cấu trúc cây quyết định")

                st.json(tree)

                st.subheader("Cây quyết định trực quan")

                fig = draw_categorical_tree(
                    tree,
                    title=title
                )

                st.pyplot(fig)

                

else:

    st.warning(
        "Vui lòng upload file CSV để bắt đầu."
    )