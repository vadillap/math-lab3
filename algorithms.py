import numpy as np
from scipy.sparse import csr_matrix


def lu(a):
    n = a.shape[0]

    l = csr_matrix(np.eye(n))
    u = csr_matrix(a.copy())
    for i in range(n - 1):
        for j in range(i + 1, n):
            l[j, i] = u[j, i] / u[i, i]

            for k in range(n):
                u[j, k] -= u[i, k] * l[j, i]
    return l, u


def gauss_forward(a, f):
    u = a.copy()
    y = f.copy()
    n = a.shape[0]

    for i in range(n - 1):
        for j in range(i + 1, n):
            s = u[j, i] / u[i, i]
            for k in range(n):
                u[j, k] -= u[i, k] * s
            y[j] -= y[i] * s

    return u, y


# для верхнетреугольной матрицы
def gauss_backward(u, y):
    n = u.shape[0]
    x = y.copy()

    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += u[i, j] * x[j]
        x[i] -= s
        x[i] /= u[i, i]

    return x


# для нижнетреугольной матрицы
def gauss_backward_lower(u, y):
    n = u.shape[0]
    x = y.copy()

    for i in range(n):
        s = 0
        for j in range(0, i):
            s += u[i, j] * x[j]
        x[i] -= s
        x[i] /= u[i, i]

    return x


def inv_lu(a):
    n = a.shape[0]

    t = csr_matrix(np.eye(n))
    l, u = lu(a)

    for i in range(n):
        t[:, i] = gauss_backward_lower(l, t[:, i])

    for i in range(n):
        t[:, i] = gauss_backward(u, t[:, i])

    return t