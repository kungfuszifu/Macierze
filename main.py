from MatrixModule import *
import matplotlib.pyplot as pl
import math
import time


def Jacobi(M, L, U, D, b):
    r = Matrix(M.shape[0], 1)
    norma, iter = [1] * 4000, 0
    negD = Mul(D, -1)
    LpU = Add(L, U)

    start = time.time()
    static = FSub(D, b)
    while norma[iter] > 1e-9 and iter < 1000:
        r = Add(FSub(negD, Mul(LpU, r)), static)
        iter += 1
        norma[iter] = norm(Sub(Mul(M, r), b))

    end = time.time()
    return iter, norma,  end - start


def Gauss(M, L, U, D, b):
    r = Matrix(M.shape[0], 1)
    norma, iter = [1] * 4000, 0
    DL = Add(D, L)
    negDL = Mul(DL, -1)

    start = time.time()
    static = FSub(DL, b)
    while norma[iter] > 1e-9 and iter < 1000:
        r = Add(FSub(negDL, Mul(U, r)), static)
        iter += 1
        norma[iter] = norm(Sub(Mul(M, r), b))

    end = time.time()
    return iter, norma, end - start


def LU(M, b):
    start = time.time()

    L, U = LUDecomposition(M)
    r = BSub(U, FSub(L, b))

    end = time.time()
    norma = norm(Sub(Mul(M, r), b))

    return norma, end - start

# H     Zad A


N = 3 * 2 * 9
a1 = 5 + 7
a2 = a3 = -1
b = Matrix(N, 1)
for i in range(N):
    b[i][0] = math.sin((i + 1) * (8 + 1))

M = Diag(N, a1, 0) + \
    Diag(N, a2, -1) + \
    Diag(N, a2, 1) + \
    Diag(N, a3, -2) + \
    Diag(N, a3, 2)

L, U, D = Matrix_to_LUD(M)

# H     Zad B


iter, norma, czas = Jacobi(M, L, U, D, b)
print("Metoda Jacobiego -> N: {0}, iteracje: {1}, czas: {2}".format(M.shape[0], iter, round(czas, 4)))

pl.plot(range(1, iter+1), norma[:iter], label="Norma błędu rezydualnego")
pl.title("Norma błędu rezydualnego w kolejnych iteracjach - Jacobi")
pl.xlabel("Iteracja")
pl.ylabel("Błąd")
pl.yscale("log")
pl.legend()
pl.show()

iter, norma, czas = Gauss(M, L, U, D, b)
print("Metoda Gaussa -> N: {0}, iteracje: {1}, czas: {2}".format(M.shape[0], iter, round(czas, 4)))
print()

pl.plot(range(1, iter+1), norma[:iter], label="Norma błędu rezydualnego")
pl.title("Norma błędu rezydualnego w kolejnych iteracjach - Gauss-Seidel")
pl.xlabel("Iteracja")
pl.ylabel("Błąd")
pl.yscale("log")
pl.legend()
pl.show()

# H     Zad C
a1 = 3
b = Matrix(N, 1)
for i in range(N):
    b[i][0] = math.sin((i + 1) * (8 + 1))

M = Diag(N, a1, 0) + \
    Diag(N, a2, -1) + \
    Diag(N, a2, 1) + \
    Diag(N, a3, -2) + \
    Diag(N, a3, 2)

L, U, D = Matrix_to_LUD(M)

iter, norma, czas = Jacobi(M, L, U, D, b)
print("Metoda Jacobiego -> N: {0}, iteracje: {1}, czas: {2}".format(M.shape[0], iter, round(czas, 4)))

pl.plot(range(1, iter+1), norma[:iter], label="Norma błędu rezydualnego")
pl.title("Norma błędu rezydualnego w kolejnych iteracjach - Jacobi")
pl.xlabel("Iteracja")
pl.ylabel("Błąd")
pl.yscale("log")
pl.legend()
pl.show()

iter, norma, czas = Gauss(M, L, U, D, b)
print("Metoda Gaussa -> N: {0}, iteracje: {1}, czas: {2}".format(M.shape[0], iter, round(czas, 4)))

pl.plot(range(1, iter+1), norma[:iter], label="Norma błędu rezydualnego")
pl.title("Norma błędu rezydualnego w kolejnych iteracjach - Gauss-Seidel")
pl.xlabel("Iteracja")
pl.ylabel("Błąd")
pl.yscale("log")
pl.legend()
pl.show()

print()

# H     Zad D

norma, czas = LU(M, b)
print("Metoda Faktoryzacji LU -> N: {0}, norma: {1}, czas: {2}".format(M.shape[0], norma, round(czas, 4)))
print()

# H     Zad E

a1 = 12
N = [100, 200, 500, 1000, 2000, 3000]
czasJ, czasG, czasLU = [], [], []

for n in N:
    b = Matrix(n, 1)
    for i in range(n):
        b[i][0] = math.sin((i + 1) * (8 + 1))

    M = Diag(n, a1, 0) + \
        Diag(n, a2, -1) + \
        Diag(n, a2, 1) + \
        Diag(n, a3, -2) + \
        Diag(n, a3, 2)

    L, U, D = Matrix_to_LUD(M)

    _, _,  czas = Jacobi(M, L, U, D, b)
    print("Metoda Jacobi -> N: {1}, czas: {0}".format(round(czas, 4), n))
    czasJ.append(czas)

    _, _, czas = Gauss(M, L, U, D, b)
    print("Metoda Gaussa-Siedla -> N: {1}, czas: {0}".format(round(czas, 4), n))
    czasG.append(czas)

    if n <= 1000:
        _, czas = LU(M, b)
        print("Metoda Faktoryzacji LU -> czas: {0}".format(round(czas, 4)))
        czasLU.append(czas)
    print()

pl.plot(N, czasJ, label="Jacobi")
pl.plot(N, czasG, label="Gauss-Seidel")
pl.plot(N[:4], czasLU, label="Faktoryzacja LU")
pl.title("Czas wykonywania algorytmów dla różnych N")
pl.xlabel("N")
pl.ylabel("Czas")
pl.legend()
pl.show()


