import numpy as np
from sympy import Matrix

P = np.array(((3, 1),
              (2, -1)))
P_inv = np.linalg.inv(P)

A = np.array(((1, 0, 0),
              (-2, 5, 3),
              (-4, -6, -4)))
EV, EVc = np.linalg.eig(A)

EVc = np.array(((1, 0, 0),
                (11, -1, 1),
                (-14, 1, -2)))

A = np.array(((-1, -3),
              (2, 2)))

M = Matrix(A)
P, J = M.jordan_form()
print(P)
print()
print(J)

# print(np.linalg.inv(A))
EV, EVc = np.linalg.eig(A)
# print(EV)
# print(EVc)
