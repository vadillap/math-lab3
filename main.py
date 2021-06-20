import sys

import numpy as np
from scipy.sparse import csr_matrix


# код с интернета, пока хз как работает, но можно юзать для дебага
# def lu1(a):
#     shape = a.shape
#
#     l, u = csr_matrix(shape), csr_matrix(shape)
#
#     for i in range(shape[0]):
#         for j in range(i, shape[1]):
#             u[i, j] = a[i, j] - (l.getrow(i) * u.getcol(j)).sum()
#             l[j, i] = (a[j, i] - (l.getrow(j) * u.getcol(i)).sum()) / u[i, i]
#
#     return l, u


def lu(a):
    n = a.shape[0]

    l = csr_matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
    u = csr_matrix(a.copy())
    for i in range(n - 1):
        for j in range(i + 1, n):
            l[j, i] = u[j, i] / u[i, i]

            for k in range(n):
                u[j, k] -= u[i, k] * l[j, i]
    return l, u


# проверяет что все угловые миноры невырождены
def check_matrix(m):
    a = m.copy()
    n = a.shape[0]
    for i in range(n - 1, -1, -1):
        if abs(np.linalg.det(a)) < sys.float_info.epsilon:
            return False
        a = np.delete(np.delete(a, i, axis=0), i, axis=1)

    return True


def test_lu():
    iterations = 40

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        print(i)

        # если матрица не поддается разлоежнию, генерируем новую
        m = csr_matrix(np.random.randint(0, 100, size=(i, i)))
        while not check_matrix(m.toarray()):
            m = csr_matrix(np.random.randint(0, 5, size=(i, i)))

        l, u = lu(m)
        m_lu = l * u

        max_delta = max(max_delta, (m_lu - m).max())

    # т.к. в процессе умножения/деления теряется точность, то выведем
    # максимальную дельту между исходной матрицей M и восстановленной L * U
    print(max_delta)


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
    n = y.shape[0]
    x = np.array([0.0 for i in range(n)])

    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += u[i, j] * x[j]
        x[i] = (y[i] - s) / u[i, i]

    return x


# для нижнетреугольной матрицы
def gauss_backward_lower(u, y):
    n = y.shape[0]
    x = np.array([0.0 for i in range(n)])

    for i in range(n):
        s = 0
        for j in range(0, i):
            s += u[i, j] * x[j]
        x[i] = (y[i] - s) / u[i, i]

    return x


def solve_with_lu(a, f):
    l, u = lu(a)

    # т.к. матрицы треугольные, можно сразу запустить обратный ход Гаусса
    y = gauss_backward_lower(l, f)
    x = gauss_backward(u, y)

    return x


def solve_with_gauss(a, f):
    u, y = gauss_forward(a.copy(), f.copy())

    return gauss_backward(u, y)


a = csr_matrix(np.array([[3.0, 2.0, -5.0], [2.0, -1.0, 3.0], [1.0, 2.0, -1.0]]))
f = np.array([-1.0, 13.0, 9.0])

l, u = lu(a.toarray())

# print(l.toarray())
# print(u.toarray())
# print((l * u).toarray())


print(solve_with_lu(a, f))
print(solve_with_gauss(a, f))
