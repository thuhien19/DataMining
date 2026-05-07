from sklearn.tree import DecisionTreeClassifier


def run_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model