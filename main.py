import sys

import numpy as np
from scipy.sparse import csr_matrix


# def lu(a):
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

    l = csr_matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
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


# m = csr_matrix(np.array([[10, -7, 0], [-3, 6, 2], [5, -1, 5]]))
# m = csr_matrix(np.array([[1, 2, 1], [2, 1, 1], [1, -1, 2]]))
# l, u = my(m)
# print(l.toarray())
# print()
# print(u.toarray())
# print()
# print(m.toarray())
# print((l * u).toarray())

# l, u = loh(m.toarray(), permute_l=True)
#
# print(l)
# print(u)
#
# l, u = lu(m)
#
# print(l.toarray())
# print()
# print(u.toarray())
# print()
# print((l * u).toarray())


test_lu()

# while True:
#     m = csr_matrix(np.random.randint(0, 5, size=(3, 3)))
#     if not check_matrix(m.toarray()):
#
#         check_matrix(m.toarray())
#         print(m.toarray())
#         print(np.linalg.det(m.toarray()))
#         break
# print(sys.float_info.epsilon)
# print(np.random.randint(0,5,size = (5,5)))
# print(check_matrix(np.array([[1,0],[0,1]])))
