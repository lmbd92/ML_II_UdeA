import numpy as np
from scipy.linalg import svd


# This fuction calculates the eigenvectors corresponding for u and v matrices
def calcMat(M, opc):
    if opc == 1:
        newM = np.dot(M.T, M)
    if opc == 2:
        newM = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eig(newM)
    ncols = np.argsort(eigenvalues)[::-1]

    if opc == 1:
        return eigenvectors[:, ncols].T
    else:
        return eigenvectors[:, ncols]


# This function calculates the eigenvalues corresponding to the sigma matrix
def calcD(M):
    if np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M)):
        newM = np.dot(M.T, M)
    else:
        newM = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eig(newM)
    eigenvalues = np.sqrt(eigenvalues)

    return eigenvalues[::-1]


# Create the matrix A
A = np.array([[4, 2, 0], [1, 5, 6]])


# Now we’ll assign values to our variables by calling the functions we created in the previous steps.

Vt = calcMat(A, 1)
U = calcMat(A, 2)
Sigma = calcD(A)

print(Vt, "\n")
print(U, "\n")
print(Sigma)

U_svd, D, VT = np.linalg.svd(A)
print(VT, "\n")
print(U_svd, "\n")
print(D)
