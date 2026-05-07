import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig