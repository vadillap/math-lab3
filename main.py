import sys
from numpy.linalg import solve
import numpy as np
from scipy.sparse import csr_matrix
from algorithms import *
from methods import *
from matrix_utils import *
from tests import *


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


a = csr_matrix(np.array([[5.0, 2.0, -1.0], [2.0, -5.0, 3.0], [1.0, 2.0, -4.0]]))
f = np.array([-1.0, 13.0, 9.0])

# a = csr_matrix(np.array([[2.0, 1.0, 1.0], [1.0, -1.0, 0.0], [3.0, -1.0, 2.0]]))
# f = np.array([2.0, -2.0, 2.0])

# l, u = lu(a.toarray())

#
# print(check_matrix(a.toarray()))
# print(l.toarray())
# print(u.toarray())
# print((l * u).toarray())


# print(solve_with_lu(a, f))
# print(solve_with_gauss(a, f))
# print(solve_with_seidel(a, f))
# t = inv_lu(a.copy())

# print((a * t).toarray())
# r = a[:, 0]
# print(r[0] + 4)

# print(gen_hilbert(3))

# test_inv()

# x = solve(a.toarray(), f)
# print(x)

# print(np.random.randint(0, 100, size=15))

test_lu_solve()
test_gauss_solve()
# test_seidel_solve()

# print(np.absolute(a).toarray())