import pandas as pd
from itertools import combinations


# ==================================================
# HÀM CHẠY ROUGH SET
# ==================================================

def run_rough_set(
    df,
    selected_attrs,
    selected_objects
):

    # ==========================================
    # CHUẨN HÓA DỮ LIỆU
    # ==========================================

    df.columns = df.columns.str.strip()

    df.index = [
        f"O{i}"
        for i in range(len(df))
    ]

    # ==========================================
    # CHUẨN HÓA TẬP X
    # ==========================================

    selected_objects = set(
        str(obj)
        for obj in selected_objects
    )

    # ==========================================
    # LOẠI BỎ CỘT ID
    # ==========================================

    ignore_cols = [
        "STT",
        "Id",
        "ID",
        "#",
        "Tên",
        "Transaction ID"
    ]

    # ==========================================
    # TOÀN BỘ THUỘC TÍNH
    # (dùng cho reduct, dependency, rules)
    # ==========================================

    all_attrs = [
        col
        for col in df.columns[:-1]
        if col not in ignore_cols
    ]

    # ==========================================
    # THUỘC TÍNH ĐƯỢC CHỌN
    # (dùng cho xấp xỉ)
    # ==========================================

    attrs = [
        col
        for col in selected_attrs
        if col not in ignore_cols
    ]

    # ==========================================
    # THUỘC TÍNH QUYẾT ĐỊNH
    # ==========================================

    decision = df.columns[-1]

    # ==========================================
    # TẬP X
    # ==========================================

    X = selected_objects

    # ==========================================
    # LỚP TƯƠNG ĐƯƠNG
    # ==========================================

    grouped = df.groupby(attrs)

    equivalence_classes = []

    for _, group in grouped:

        eq_class = set(group.index)

        equivalence_classes.append(eq_class)

    # ==========================================
    # LOWER APPROXIMATION
    # ==========================================

    lower = set()

    for eq_class in equivalence_classes:

        if eq_class.issubset(X):

            lower |= eq_class

    # ==========================================
    # UPPER APPROXIMATION
    # ==========================================

    upper = set()

    for eq_class in equivalence_classes:

        if eq_class & X:

            upper |= eq_class

    # ==========================================
    # ĐỘ CHÍNH XÁC
    # ==========================================

    accuracy = 0

    if len(upper) > 0:

        accuracy = round(
            len(lower) / len(upper),
            3
        )

    # ==========================================
    # POSITIVE REGION
    # ==========================================

    grouped_all = df.groupby(all_attrs)

    positive = []

    for _, group in grouped_all:

        if len(group[decision].unique()) == 1:

            positive.extend(
                group.index.tolist()
            )

    dependency = round(
        len(positive) / len(df),
        3
    )

     # ==========================================
    # MA TRẬN PHÂN BIỆT
    # ==========================================

    disc = []

    for i in range(len(df)):

        for j in range(i):

            # khác quyết định
            if df.iloc[i][decision] != df.iloc[j][decision]:

                diff = []

                for a in all_attrs:

                    if df.iloc[i][a] != df.iloc[j][a]:

                        diff.append(a)

                if diff:

                    disc.append(set(diff))

    # ==========================================
    # KIỂM TRA REDUCT
    # ==========================================

    def is_red(subset, disc):

        subset = set(subset)

        for d in disc:

            # phải giao với mọi tập phân biệt
            if subset.isdisjoint(d):

                return False

        return True

    # ==========================================
    # TÌM REDUCT TỐI THIỂU
    # ==========================================

    reducts = []

    for r in range(1, len(all_attrs) + 1):

        for sub in combinations(all_attrs, r):

            sub = set(sub)

            if is_red(sub, disc):

                minimal = True

                # kiểm tra tối thiểu
                for k in range(1, len(sub)):

                    for s in combinations(sub, k):

                        if is_red(set(s), disc):

                            minimal = False
                            break

                    if not minimal:
                        break

                if minimal:

                    reducts.append(sub)

    # ==========================================
    # SINH LUẬT PHÂN LỚP
    # ==========================================

    rules = []

    for reduct in reducts:

        reduct_list = list(reduct)

        grouped_rules = df.groupby(reduct_list)

        for name, group in grouped_rules:

            # lớp quyết định duy nhất
            if len(group[decision].unique()) == 1:

                values = (
                    name
                    if isinstance(name, tuple)
                    else (name,)
                )

                cond = [
                    f"{attr}={val}"
                    for attr, val in zip(
                        reduct_list,
                        values
                    )
                ]

                rule = (
                    f"IF {' AND '.join(cond)} "
                    f"THEN {decision}="
                    f"{group[decision].iloc[0]}"
                )

                rules.append(rule)

    # ==========================================
    # RETURN
    # ==========================================

    return (
        equivalence_classes,
        lower,
        upper,
        accuracy,
        dependency,
        reducts,
        rules[:5]
    )