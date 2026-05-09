import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin

from utils.preprocessing import basic_preprocessing, encode_data


def run_kmeans(df, k):
    data = basic_preprocessing(df)

    encoded_data, encoders = encode_data(data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(encoded_data)

    np.random.seed(42)

    random_idx = np.random.choice(
        len(scaled_data),
        k,
        replace=False
    )

    initial_centroids = scaled_data[random_idx]

    # phân cụm ban đầu theo centroid ban đầu
    initial_clusters = pairwise_distances_argmin(
        scaled_data,
        initial_centroids
    )

    initial_result = data.copy()
    initial_result["Initial Cluster"] = initial_clusters

    model = KMeans(
        n_clusters=k,
        init=initial_centroids,
        n_init=1,
        random_state=42
    )

    final_clusters = model.fit_predict(scaled_data)

    final_result = data.copy()
    final_result["Final Cluster"] = final_clusters

    encoded_result = encoded_data.copy()
    encoded_result["Initial Cluster"] = initial_clusters
    encoded_result["Final Cluster"] = final_clusters

    return (
        model,
        initial_result,
        final_result,
        encoded_result,
        scaled_data,
        initial_centroids
    )


def draw_kmeans_cluster(scaled_data, clusters, title="Biểu đồ gom cụm K-means"):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    plot_df = pd.DataFrame({
        "PC1": pca_data[:, 0],
        "PC2": pca_data[:, 1],
        "Cluster": clusters
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == cluster]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster}",
            s=100
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)

    return fig