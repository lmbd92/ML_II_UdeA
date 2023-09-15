""" Punto 5.  expeiment with the silhouette score to determine the optimal number of clusters for the scattered data X, using the k-means and k-medoids algorithms."""

from mypackage.k_means import KMeans
from mypackage.k_medoids import KMedoids
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
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


plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scattered Data")
plt.show()


# Initialize a list to store silhouette scores for k-means and k-medoids
kmeans_scores = []
kmedoids_scores = []

for k in range(1, 6):
    # For k-means
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X)
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_scores.append(kmeans_silhouette)

    # For k-medoids
    kmedoids = KMedoids(n_clusters=k)
    kmedoids.fit(X)
    kmedoids_labels = kmedoids.predict(X)
    kmedoids_silhouette = silhouette_score(X, kmedoids_labels)
    kmedoids_scores.append(kmedoids_silhouette)

# Plot the silhouette scores for both k-means and k-medoids
plt.plot(range(1, 6), kmeans_scores, label="K-Means")
plt.plot(range(1, 6), kmedoids_scores, label="K-Medoids")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.show()
