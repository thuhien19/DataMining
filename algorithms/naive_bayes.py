from sklearn.naive_bayes import CategoricalNB


def run_naive_bayes(X_train, y_train):
    model = CategoricalNB()
    model.fit(X_train, y_train)
    return model