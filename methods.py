import numpy as np

from algorithms import *


def solve_with_lu(a, f):
    l, u = lu(a)

    # т.к. матрицы треугольные, можно сразу запустить обратный ход Гаусса
    y = gauss_backward_lower(l, f)
    x = gauss_backward(u, y)

    return x


def solve_with_gauss(a, f):
    u, y = gauss_forward(a.copy(), f.copy())

    return gauss_backward(u, y)


def solve_with_seidel(A, b):
    return seidel(A, b, 1e-9)


def seidel(A, b, tol):
    N = A.shape[0]
    maxIterations = 100
    x = [1.0 for i in range(N)]
    xprev = [0.0 for i in range(N)]
    for i in range(maxIterations):
        for j in range(N):
            xprev[j] = x[j]
        for j in range(N):
            summ = 0.0
            for k in range(N):
                if (k != j):
                    summ = summ + A[j, k] * x[k]
            x[j] = (b[j] - summ) / A[j, j]
        # print(x)
        diff1norm = 0.0
        oldnorm = 0.0
        for j in range(N):
            diff1norm = diff1norm + abs(x[j] - xprev[j])
            oldnorm = oldnorm + abs(xprev[j])
        if oldnorm == 0.0:
            oldnorm = 1.0
        norm = diff1norm / oldnorm
        if (norm < tol) and i != 0:
            print("Sequence converges to ", end="")
            print(x, end="")
            print(". Took", i + 1, "iterations.")
            return x
    raise Exception("Does not coverage")


def solve_with_yakobi(a, f):
    # предподсчет матрицы D и обратной к ней
    d = csr_matrix(a.shape)
    d.setdiag(a.diagonal(0), 0)
    d_inv = d.copy()
    d_inv = d_inv.power(-1)

    # основные матрица B и вектор g, которые используются в шаге
    b = d_inv * (d - a)
    g = d_inv * f

    x = f.copy()
    iterations = 100000
    epsilon = 1e-9

    # выполняем пока не закончились операции или не выполнилось условие останова
    # используем условие останова, которое не требует вычисление нормы матрицы
    # саму норму берем как ||X||∞, что по сути есть максимум из вектора
    while iterations > 0 and np.absolute(a * x - f).max() > epsilon:
        x = b * x + g
        iterations -= 1

    return x
