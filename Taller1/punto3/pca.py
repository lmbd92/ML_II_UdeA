# To implement PCA from scratch, we'll need to perform the following steps:

""" 
1. Mean-center the data.
2. Calculate the covariance matrix of the mean-centered data.
3. Compute the eigenvectors and eigenvalues of the covariance matrix.
4. Sort the eigenvectors by their corresponding eigenvalues in descending order.
5. Select the top n_components eigenvectors as the principal components.
6. Transform the data using the selected principal components. """


import logging
import numpy as np

from punto3.base import BaseEstimator

np.random.seed(1000)


class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components):
        """Principal component analysis (PCA) implementation.

        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.

        Parameters
        ----------
        n_components : int
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)

    def _decompose(self, X):
        # Mean centering
        X = X.copy()
        X -= self.mean

        # Calculate the covariance matrix
        covariance_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

        # Explained variance ratio
        explained_variance_ratio = eigenvalues[sorted_indices] / \
            eigenvalues.sum()
        logging.info("Explained variance ratio: %s" %
                     (explained_variance_ratio[:self.n_components]))

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components)

    def _predict(self, X=None):
        return self.transform(X)
