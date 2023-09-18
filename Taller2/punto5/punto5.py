""" Punto 5.  experiment with the silhouette score to determine the optimal number of clusters for the scattered data X, using the k-means and k-medoids algorithms."""

from mymlpackage import KMeans, KMedoids
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt


X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)

x_list = X.tolist()


plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scattered Data")
plt.show()


# Initialize a list to store silhouette scores for k-means and k-medoids
k_values = range(1, 6)
kmeans_scores = []
kmedoids_scores = []

for k in k_values:
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X)

    # Check the number of unique labels
    unique_labels_kmeans = np.unique(kmeans_labels)
    if len(unique_labels_kmeans) < 2:
        print(
            f"K-means with {k} clusters produced only {len(unique_labels_kmeans)} unique label(s). Skipping."
        )
        continue

    # Calculate silhouette score
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_scores.append(kmeans_silhouette)

    # Perform K-medoids clustering
    kmedoids = KMedoids(n_clusters=k)
    kmedoids.fit(x_list)
    kmedoids_labels = kmedoids.predict(x_list)

    # Check the number of unique labels
    unique_labels_kmedoids = np.unique(kmedoids_labels)
    if len(unique_labels_kmedoids) < 2:
        print(
            f"K-medoids with {k} clusters produced only {len(unique_labels_kmedoids)} unique label(s). Skipping."
        )
        continue

    # Calculate silhouette score
    kmedoids_silhouette = silhouette_score(x_list, kmedoids_labels)
    kmedoids_scores.append(kmedoids_silhouette)

# Plot the silhouette scores for both k-means and k-medoids
plt.plot(k_values, kmeans_scores, label="K-Means")
plt.plot(k_values, kmedoids_scores, label="K-Medoids")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()

best_k_kmeans = k_values[np.argmax(kmeans_scores)]
best_score_kmeans = max(kmeans_scores)

best_k_kmedoids = k_values[np.argmax(kmedoids_scores)]
best_score_kmedoids = max(kmedoids_scores)

print(f"Best K for K-means: {best_k_kmeans}, Silhouette Score: {best_score_kmeans}")
print(
    f"Best K for K-medoids: {best_k_kmedoids}, Silhouette Score: {best_score_kmedoids}"
)
