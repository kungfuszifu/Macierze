import copy
import math
import numpy as np


def Matrix(rows, cols, value=0):
    return np.array([[float(value)] * cols for _ in range(rows)])


def Diag(N, value, axis=0):
    x = Matrix(N, N)
    for r in range(N):
        for c in range(N):
            if r + axis == c:
                x[r, c] = value
    return x


def inv_diagonal(D):
    x = Matrix(D.shape[0], D.shape[1])
    for r in range(D.shape[0]):
        for c in range(D.shape[1]):
            if c == r:
                x[r, c] = 1 / D[r, c]
    return x


def Add(L, R):
    if [L.shape[0], L.shape[1]] == [R.shape[0], R.shape[1]]:
        x = Matrix(L.shape[0], L.shape[1])
        for r in range(L.shape[0]):
            for c in range(L.shape[1]):
                x[r, c] = L[r, c] + R[r, c]
        return x


def Sub(L, R):
    if [L.shape[0], L.shape[1]] == [R.shape[0], R.shape[1]]:
        x = Matrix(L.shape[0], L.shape[1])
        for r in range(L.shape[0]):
            for c in range(L.shape[1]):
                x[r, c] = L[r, c] - R[r, c]
        return x


def Mul(L, R):
    if isinstance(R, int):
        return np.array([[a * R for a in row] for row in L])

    if L.shape[1] == R.shape[0]:
        x = Matrix(L.shape[0], R.shape[1])
        for r in range(L.shape[0]):
            for c in range(R.shape[1]):
                x[r, c] = sum(np.multiply(L[r, :], R[:, c]))
        return x


def Matrix_to_LUD(M):
    L = Matrix(M.shape[0], M.shape[1])
    U = Matrix(M.shape[0], M.shape[1])
    D = Matrix(M.shape[0], M.shape[1])

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if r < c:
                U[r, c] = M[r, c]
            if r > c:
                L[r, c] = M[r, c]
            if r == c:
                D[r, c] = M[r, c]
    return L, U, D


def FSub(L, b):
    x = Matrix(L.shape[0], 1)
    for i in range(L.shape[1]):
        tmp = b[i, 0] - sum(np.multiply(L[i, 0:i], x[0:i, 0]))
        x[i, 0] = tmp / L[i, i]
    return x


def BSub(U, b):
    x = Matrix(U.shape[0], 1)
    for i in range(U.shape[0] - 1, -1, -1):
        tmp = b[i][0] - sum(np.multiply(U[i, i+1:U.shape[0]], x[i+1:U.shape[0], 0]))
        x[i][0] = tmp / U[i][i]
    return x


def LUDecomposition(M):
    N = len(M)
    L = Diag(N, 1)
    U = copy.deepcopy(M)

    for i in range(N - 1):  # kolumna
        for j in range(i + 1, N):  # wiersz
            L[j, i] = U[j, i] / U[i, i]

            for k in range(i, N):
                U[j, k] = U[j, k] - L[j, i] * U[i, k]

    return L, U


def norm(M):
    suma = 0
    for i in range(M.shape[0]):
        suma += M[i][0] ** 2
    return math.sqrt(suma)
