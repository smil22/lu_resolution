import numpy as np
import copy as cp

def lu_resolution(A,b):
    """We resolve the system Ax=b by LU decomposition.
    A = LU and yields LUX = b. Then we consider that UX = Y.
    """
    n = np.size(A,1)
    L = np.eye(n)
    U = cp.copy(A)
    #Décomposition LU
    for j in range(n):
        for i in range(j+1,n):
            coeff = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - coeff *  U[j,:]
            L[i,j] = coeff
    #Calcul de y (Système triangulaire inférieure)
    y = np.zeros((n,1))
    for i in range(n):
        y[i] = (b[i] - np.sum([L[i,j] * y[j] for j in range(i)])) / L[i,i]
    
    #Calcul de x (Système triangulaire supérieur)
    x = np.zeros((n,1))
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.sum([U[i,j] * x[j] for j in range(n-1,i,-1)])) / U[i,i]
    return L,U,y,x
