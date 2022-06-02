#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:28:16 2022

@author: kissami
"""

import numpy as np
import kronecker_csr as core_csr


def kron_product_csr(A, x):
    
        
    if len(A) == 2:
        A1, A2 = A 
        n_rows_1 = A1.shape[0]
        n_cols_1 = A1.shape[1]
        n_rows_2 = A2.shape[0]
        n_cols_2 = A2.shape[1]
    
        n = max(n_rows_1, max(n_cols_1, max(n_rows_2, n_cols_2)))
        
        W1 = np.zeros((n, n))
        W2 = np.zeros((n, n))
        
        nrows = n_rows_1 * n_rows_2
        y = np.empty(nrows)
        
        core_csr.kron_2d(A1.data, A1.indices, A1.indptr,
                         A2.data, A2.indices, A2.indptr,
                         n_rows_1, n_cols_1,
                         n_rows_2, n_cols_2,
                         x,
                         W1, W2,
                         y)
        return y
    elif len(A) == 3:
        
        A1, A2, A3 = A 
        
        n_rows_1 = A1.shape[0]
        n_cols_1 = A1.shape[1]
        n_rows_2 = A2.shape[0]
        n_cols_2 = A2.shape[1]
        n_rows_3 = A3.shape[0]
        n_cols_3 = A3.shape[1]
        
        
        n_rows_12 = n_rows_1 * n_rows_2
        n_cols_12 = n_cols_1 * n_cols_2
    
        nrows = n_rows_1 * n_rows_2 * n_rows_3
        
        y = np.empty(nrows)
        
        
        n = max(n_rows_12, max(n_cols_12, max(n_rows_3, n_cols_3)))
        W1 = np.empty((n, n))
        W2 = np.empty((n, n))
    
        n = max(n_rows_1, max(n_cols_1, max(n_rows_2, n_cols_2)))
        W3 = np.empty((n, n))
        W4 = np.empty((n, n))
    
        core_csr.kron_3d(A1.data, A1.indices, A1.indptr,
                         A2.data, A2.indices, A2.indptr,
                         A3.data, A3.indices, A3.indptr,
                         n_rows_1, n_cols_1,
                         n_rows_2, n_cols_2,
                         n_rows_3, n_cols_3,
                         x,
                         W1, W2, W3, W4,
                         y )

        return y


from scipy.sparse import kron as sp_kron
from scipy.sparse import csr_matrix

#######################**********2D*******#####################################
rA = 3
cA = 6

rB = 40
cB = 9

A = np.random.rand(rA, cA)
B = np.random.rand(rB, cB)

D = sp_kron(A, B)

(rD, cD) = D.shape

x = np.random.rand(cD)

y = D.dot(x)

xbar = x.reshape((cA,cB))

#AkronB = AXBT
yb1 = A.dot(xbar)
yb2 = yb1.dot(B.T)
yb2 = yb2.reshape(rA*rB)

print("dot product:", max(np.fabs(y-yb2)))


A = csr_matrix(A)
B = csr_matrix(B)
A = (A, B)
yb3 = kron_product_csr(A, x)

print("kron product csr:", max(np.fabs(y-yb3)))
###############################################################################

#######################**********3D*******#####################################
rA = 3
cA = 6

rB = 40
cB = 9

rC = 3
cC = 10

A = np.random.rand(rA, cA)
B = np.random.rand(rB, cB)
C = np.random.rand(rC, cC)

D = sp_kron(A, sp_kron(B,C))

(rD, cD) = D.shape

x = np.random.rand(cD)

y = D.dot(x)

#xbar = x.reshape((cA,cB))

##AkronB = AXBT
#yb1 = A.dot(xbar)
#yb2 = yb1.dot(B.T)
#yb2 = yb2.reshape(rA*rB)
#
#print("dot product:", max(np.fabs(y-yb2)))
#
#
A = csr_matrix(A)
B = csr_matrix(B)
C = csr_matrix(C)

A = (A, B, C)
yb3 = kron_product_csr(A, x)
#
print("kron product csr:", max(np.fabs(y-yb3)))
###############################################################################









