import sys
import numpy as np
from scipy.sparse import csr_matrix
from algorithms import lu
from methods import *
from matrix_utils import *


def test_lu():
    iterations = 40

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        m = gen_random(i)

        l, u = lu(m)
        m_lu = l * u

        max_delta = max(max_delta, np.absolute(m_lu - m).max())

    # т.к. в процессе умножения/деления теряется точность, то выведем
    # максимальную дельту между исходной матрицей M и восстановленной L * U
    print(max_delta)


def test_inv():
    iterations = 15

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        print(i)
        m = gen_random(i)

        t = inv_lu(m.copy())
        e = m * t

        e_ex = csr_matrix(np.eye(i))

        max_delta = max(max_delta, np.absolute(e - e_ex).max())

    # т.к. в процессе умножения/деления теряется точность, то выведем
    # максимальную дельту между исходной матрицей M и восстановленной L * U
    print(max_delta)


def test_solve(method):
    iterations = 15

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        print(i, end=" ")
        a = gen_random(i)
        x = np.random.randint(0, 100, size=i).astype(float)

        f = a * x

        x_solved = method(a, f)

        max_delta = max(max_delta, np.absolute(x_solved - x).max())

    print()
    print(max_delta)


def test_lu_solve():
    test_solve(solve_with_lu)


def test_gauss_solve():
    test_solve(solve_with_gauss)


def test_seidel_solve():
    iterations = 15

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        print(i, end=" ")
        while True:
            a = get_random_dominant(i)
            x = np.random.randint(0, 100, size=i).astype(float)
            f = a * x

            try:
                x_solved = solve_with_seidel(a, f)
                max_delta = max(max_delta, np.absolute(x_solved - x).max())
                break
            except Exception:
                pass

    print()
    print(max_delta)
