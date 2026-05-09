from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def run_decision_tree(X, y, criterion="entropy"):
    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )

    model.fit(X, y)

    return model


def draw_decision_tree(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(18, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )

    return fig