import library
import numpy as np
import scipy.linalg as la

n = 3
A = np.random.randn(n,n)
sol = np.random.randn(n,1)
b = A.dot(sol)

L,U,y,x = library.lu_resolution(A, b)

print('Residual: {0:3.4e}'.format(la.norm(A.dot(x) - b)))
