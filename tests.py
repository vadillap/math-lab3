import sys
import numpy as np
from scipy.sparse import csr_matrix
from algorithms import lu
from matplotlib import pyplot as plt
import time
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
    iterations = 100

    max_delta = sys.float_info.epsilon
    for i in range(1, iterations):
        print(i, end=" ")
        while True:
            a = get_random_dominant(i)
            x = np.random.randint(0, 100, size=i).astype(float)
            f = a * x

            try:
                x_solved = yakobi(a, f)
                max_delta = max(max_delta, np.absolute(x_solved - x).max())
                break
            except Exception:
                pass

    print()
    print(max_delta)


def compare_methods():
    n_arr = list(range(10, 100, 10))  # 10**3, 10**4, 10**5, 10**6]

    t1 = []
    t2 = []
    for n in n_arr:
        m = get_random_dominant(n)
        x = np.random.randint(0, 50, size=n).astype(float)
        f = m * x

        print("Прямой метод")
        t = time.time_ns()
        print(solve_with_gauss(m, f))
        t1.append((time.time_ns() - t) / 1e9)

        print("Итерационный метод")
        t = time.time_ns()
        print(solve_with_yakobi(m, f))
        t2.append((time.time_ns() - t) / 1e9)

    plt.figure(dpi=500)
    plt.plot(n_arr, t1, label="LU-разложение")
    plt.plot(n_arr, t2, label="Якоби")
    plt.xlabel("Размерность n")
    plt.ylabel("Время t, с")
    plt.grid()
    plt.legend()
    plt.title("Сравнение методов на малых размерностях")
    plt.show()

    n_arr_big = list(range(10**2, 10**4, (10**4 - 10**2) // 15))
    for n in n_arr_big:
        print(n)
        m = get_random_sparse(n)
        x = np.random.randint(0, 50, size=n).astype(float)
        f = m * x

        print("Итерационный метод")
        t = time.time_ns()
        print(solve_with_yakobi(m, f))
        t2.append((time.time_ns() - t) / 1e9)

    plt.figure(dpi=500)
    plt.plot(n_arr, t1, label="LU-разложение")
    plt.plot(n_arr + n_arr_big, t2, label="Якоби")
    plt.xlabel("Размерность n")
    plt.ylabel("Время t, с")
    plt.grid()
    plt.legend()
    plt.title("Сравнение методов на больших размерностях")
    plt.show()
