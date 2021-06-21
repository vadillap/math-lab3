import random

import numpy as np
import sys

import scipy.sparse
from scipy.sparse import csr_matrix


# проверяет что все угловые миноры невырождены
def check_matrix(m):
    a = m.copy()
    n = a.shape[0]
    for i in range(n - 1, -1, -1):
        if abs(np.linalg.det(a)) < sys.float_info.epsilon:
            return False
        a = np.delete(np.delete(a, i, axis=0), i, axis=1)

    return True


def check_diag_dominant(m):
    d = np.diag(np.abs(m))
    s = np.sum(np.abs(m), axis=1) - d
    return np.all(d > s)


def gen_hilbert(k):
    h = np.empty((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            h[i, j] = 1 / (i + j + 1)

    return csr_matrix(h)


def gen_random(k):
    while True:
        m = np.random.randint(0, 100, size=(k, k))
        if check_matrix(m):
            return csr_matrix(m.astype(float))


def get_random_dominant(k):
    while True:
        m = np.random.randint(0, 100, size=(k, k))
        m += np.diagflat(np.random.randint(100, 200, size=k) * k)
        if check_diag_dominant(m):
            return csr_matrix(m.astype(float))

        print("nope")


def get_random_sparse(k):
    while True:
        m = scipy.sparse.rand(k, k, format="csr")
        m += np.diagflat(np.random.randint(100, 200, size=k) * k)
        if check_diag_dominant(m):
            return csr_matrix(m.astype(float))

        print("nope")


def get_diagonal(n, k):
    A = csr_matrix((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] = random.choice([-4.0, -3.0, -2.0, -1.0, 0.0])
            else:
                A[i, j] = 0.0

    A_sums = A.sum(axis=1)  # сумма строк

    for i in range(n):
        A[i, i] = -A_sums[i] + 10.0 ** (-k)

    return A
