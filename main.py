import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import lu as loh


def lu(a):
    shape = a.shape

    l, u = csr_matrix(shape), csr_matrix(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            u[0, i] = a[0, i]
            l[i, 0] = a[i, 0] / u[0, 0]

            u[i, j] = a[i, j] - (l.getrow(i) * u.getcol(j)).sum()

            if i >= j:
                l[j, i] = 0

            else:
                l[j, i] = (a[j, i] - (l.getrow(j) * u.getcol(i)).sum()) / u[i, i]

    return l, u



# m = csr_matrix(np.array([[10, -7, 0], [-3, 6, 2], [5, -1, 5]]))
m = csr_matrix(np.array([[1, 2, 1], [2, 1, 1], [1, -1, 2]]))




# l, u = loh(m.toarray(), permute_l=True)
#
# print(l)
# print(u)
#
l, u = lu(m)
#
print(l.toarray())
print()
print(u.toarray())
print()
print((l * u).toarray())