import pandas as pd

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

from mlxtend.preprocessing import TransactionEncoder


def run_apriori(
    data,
    min_support=0.3,
    min_confidence=0.6
):

    # =========================
    # CHUYỂN DỮ LIỆU THÀNH TRANSACTIONS
    # =========================

    transactions = []

    for _, row in data.iterrows():

        transaction = []

        for col in data.columns:

            item = f"{col}={row[col]}"

            transaction.append(item)

        transactions.append(transaction)

    # =========================
    # ENCODE DỮ LIỆU
    # =========================

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    # =========================
    # TÌM TẬP PHỔ BIẾN
    # =========================

    frequent_itemsets = apriori(
        df,
        min_support=min_support,
        use_colnames=True
    )

    # =========================
    # TÌM LUẬT KẾT HỢP
    # =========================

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    return (
        frequent_itemsets,
        rules,
        df
    )