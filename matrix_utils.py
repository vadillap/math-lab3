import numpy as np
import sys
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


def gen_hilbert(k):
    h = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            h[i, j] = 1 / (i + j + 1)

    return h


def gen_random(k):
    while True:
        m = np.random.randint(0, 100, size=(k, k))
        if check_matrix(m):
            return csr_matrix(m.astype(float))