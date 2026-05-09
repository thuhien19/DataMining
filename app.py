import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from utils.evaluation import evaluate_model
from utils.visualization import plot_confusion_matrix
from utils.preprocessing import (
    preprocess_classification_data,
    basic_preprocessing, 
    preprocess_scaled_data
)

from algorithms.naive_bayes import naive_bayes_predict
from algorithms.correlation import run_correlation
from algorithms.apriori import run_apriori
from algorithms.rough_set import run_rough_set

from algorithms.decision_tree import (
    build_tree,
    draw_categorical_tree
)
from algorithms.logistic_regression import run_logistic_regression

from algorithms.random_forest import run_random_forest

from prediction.bank_prediction import (
    predict_new_data
)


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
        "Rough Set",
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

    df = basic_preprocessing(df)

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

            # MA TRẬN TƯƠNG QUAN

            st.subheader(
                "MA TRẬN TƯƠNG QUAN"
            )

            st.dataframe(correlation_matrix)

            # HEATMAP

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
    # ROUGH SET
    # ==================================================

    elif algorithm == "Rough Set":

        st.subheader("CHỌN THUỘC TÍNH")

        attrs = list(df.columns[:-1])

        selected_attrs = st.multiselect(
            "Chọn thuộc tính B",
            attrs,
            default=attrs[:2]
        )

        # ==========================================
        # OBJECT IDs
        # ==========================================

        object_ids = [
            f"O{i}"
            for i in range(len(df))
        ]

        selected_objects = st.multiselect(
            "Chọn tập X (dòng dữ liệu)",
            object_ids,
            default=object_ids[:3]
        )

        if st.button(
            "Chạy thuật toán",
            key="roughset_btn"
        ):

            (
                equivalence_classes,
                lower,
                upper,
                accuracy,
                dependency,
                reducts,
                rules
            ) = run_rough_set(
                df,
                selected_attrs,
                selected_objects
            )

            st.success(
                "Chạy Rough Set thành công!"
            )

            # ==========================================
            # LỚP TƯƠNG ĐƯƠNG
            # ==========================================

            st.subheader(
                "LỚP TƯƠNG ĐƯƠNG"
            )

            for i, eq in enumerate(
                equivalence_classes,
                1
            ):

                st.write(
                    f"Lớp {i}: {sorted(eq)}"
                )

            # ==========================================
            # LOWER APPROXIMATION
            # ==========================================

            st.subheader(
                "XẤP XỈ DƯỚI"
            )

            st.success(
                f"Lower(B,X) = {sorted(lower)}"
            )

            # ==========================================
            # UPPER APPROXIMATION
            # ==========================================

            st.subheader(
                "XẤP XỈ TRÊN"
            )

            st.info(
                f"Upper(B,X) = {sorted(upper)}"
            )

            # ==========================================
            # ĐỘ CHÍNH XÁC
            # ==========================================

            st.subheader(
                "ĐỘ CHÍNH XÁC ROUGH SET"
            )

            st.metric(
                "Accuracy",
                accuracy
            )

            # ==========================================
            # DEPENDENCY
            # ==========================================

            st.subheader(
                "ĐỘ PHỤ THUỘC"
            )

            st.metric(
                "Dependency",
                dependency
            )

            # ==========================================
            # REDUCT
            # ==========================================

            st.subheader(
                "CÁC REDUCT"
            )

            reduct_df = pd.DataFrame({
                "Reduct": [
                    ", ".join(sorted(r))
                    for r in reducts
                ]
            })

            st.dataframe(reduct_df)

            # ==========================================
            # RULES
            # ==========================================

            st.subheader(
                "LUẬT PHÂN LỚP"
            )

            for i, rule in enumerate(
                rules,
                1
            ):

                st.write(
                    f"Luật {i}: {rule}"
                )
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

                

        # ==================================================
        # LOGISTIC REGRESSION
        # ==================================================

        elif algorithm == "Logistic Regression":

            st.subheader(
                "LOGISTIC REGRESSION"
            )

            (
                X_train,
                X_test,
                y_train,
                y_test,
                target_col,
                feature_cols,
                encoders,
                original_data,
                encoded_data
            ) = preprocess_scaled_data(df)

            # ==========================================
            # TRAIN MODEL
            # ==========================================

            if st.button(
                "Huấn luyện Logistic Regression",
                key="lr_btn"
            ):

                st.session_state["lr_model"] = run_logistic_regression(
                    X_train,
                    y_train
                )

                model = st.session_state["lr_model"]

                st.success(
                    "Huấn luyện Logistic Regression thành công!"
                )

                # ==========================================
                # DỰ ĐOÁN TEST
                # ==========================================

                y_pred = model.predict(
                    X_test
                )

                y_true_label = encoders[
                    target_col
                ].inverse_transform(y_test)

                y_pred_label = encoders[
                    target_col
                ].inverse_transform(y_pred)

                result_df = pd.DataFrame({
                    "Thực tế": y_true_label,
                    "Dự đoán": y_pred_label
                })

                st.subheader(
                    "KẾT QUẢ DỰ ĐOÁN"
                )

                st.dataframe(result_df)

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                st.subheader(
                    "ĐỘ CHÍNH XÁC"
                )

                st.metric(
                    "Accuracy",
                    round(accuracy, 4)
                )

                report = classification_report(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                st.subheader(
                    "CLASSIFICATION REPORT"
                )

                st.text(report)

                matrix = confusion_matrix(
                    y_test,
                    y_pred
                )

                st.subheader(
                    "CONFUSION MATRIX"
                )

                fig_cm = plot_confusion_matrix(
                    matrix
                )

                st.pyplot(fig_cm)

            # ==========================================
            # UPLOAD FILE PREDICT
            # ==========================================

            st.divider()

            st.subheader(
                "DỰ ĐOÁN KHÁCH HÀNG TỪ FILE CSV"
            )

            predict_file = st.file_uploader(
                "Upload file khách hàng mới",
                type=["csv"],
                key="lr_predict_file"
            )

            # ==========================================
            # KIỂM TRA MODEL
            # ==========================================

            if "lr_model" not in st.session_state:

                st.warning(
                    "Vui lòng huấn luyện Logistic Regression trước."
                )

            else:

                if predict_file is not None:

                    predict_df = pd.read_csv(
                        predict_file
                    )

                    st.subheader(
                        "Dữ liệu cần dự đoán"
                    )

                    st.dataframe(predict_df)

                    # ==========================================
                    # BUTTON PREDICT
                    # ==========================================

                    if st.button(
                        "Dự đoán khách hàng",
                        key="predict_lr_btn"
                    ):

                        result_df = predict_new_data(
                            predict_df,
                            st.session_state["lr_model"],
                            feature_cols,
                            encoders,
                            target_col,
                            encoded_data
                        )

                        st.subheader(
                            "KẾT QUẢ DỰ ĐOÁN"
                        )

                        st.dataframe(result_df)

                        # ==========================================
                        # DOWNLOAD CSV
                        # ==========================================

                        csv = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Tải file kết quả",
                            csv,
                            "prediction_result.csv",
                            "text/csv"
                        )
else:

    st.warning(
        "Vui lòng upload file CSV để bắt đầu."
    )

