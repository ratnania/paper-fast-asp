#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from numpy import linalg as la

from scipy.sparse.linalg import spsolve_triangular, spsolve
from scipy.sparse.linalg import eigs
from scipy.sparse import identity, diags, csr_matrix, bmat

from utulities import Gauss_Seidel, succesive_over_relaxation, Gauss_Seidel_block_1, Gauss_Seidel_block_2

from tabulate import tabulate

from matrices_3d import *

from kroneker import *


p1 = 2
p2 = 2
p3 = 2
n1 = 2**2
n2 = 2**2
n3 = 2**2

p  = (p1, p2, p3)
n  = (n1, n2, n3)

T1 = knot_vector(n1, p1)
T2 = knot_vector(n2, p2)
T3 = knot_vector(n3, p3)
T  = (T1, T2, T3)

tau = 0.
normalize = True

matrices, confficients = curl_matrix(T         = T, 
                                     p         = p, 
                                     tau       = tau,
                                     normalize = normalize, 
                                     form      = 'csr')

A11 = matrices[0] 
A12 = matrices[1] 
A13 = matrices[2] 
A21 = matrices[3] 
A22 = matrices[4] 
A23 = matrices[5] 
A31 = matrices[6] 
A32 = matrices[7] 
A33 = matrices[8] 

c11 = confficients[0] 
c12 = confficients[1] 
c13 = confficients[2] 
c21 = confficients[3] 
c22 = confficients[4] 
c23 = confficients[5] 
c31 = confficients[6] 
c32 = confficients[7] 
c33 = confficients[8] 
 
 

(D1, M2, K3, D1, K2, M3, D1, M2, M3) = A11
(R1, R2_T, M3)                       = A12
(R1, M2, R3_T)                       = A13
(R1_T, R2, M3_T)                     = A21
(K1, D2, M3, M1, D2, K3, M1, D2, M3) = A22
(M1, R2 , R3_T)                      = A23
(R1_T, M2, R3)                       = A31
(M1, R2_T, R3)                       = A32
(M1, K2, D3, K1, M2, D3, M1, M2, D3) = A33

M11 = sp_kron(sp_kron(D1,M2),K3) + sp_kron(sp_kron(D1,K2),M3) + tau*sp_kron(sp_kron(D1,M2),M3)
M12 = -sp_kron(sp_kron(R1,R2_T),M3)
M13 = -sp_kron(sp_kron(R1,M2),R3_T)
M21 = -sp_kron(sp_kron(R1_T,R2),M3_T)
M22 = sp_kron(sp_kron(K1,D2),M3)+sp_kron(sp_kron(M1,D2),K3) + tau*sp_kron(sp_kron(M1,D2),M3) 
M23 = -sp_kron(sp_kron(M1,R2),R3_T)
M31 = -sp_kron(sp_kron(R1_T,M2),R3)
M32 = -sp_kron(sp_kron(M1,R2_T),R3)
M33 = sp_kron(sp_kron(M1,K2),D3) + sp_kron(sp_kron(K1,M2),D3) + tau*sp_kron(sp_kron(M1,M2),D3)  



print('=============== shape of elementary matrices ===============')
print(np.shape(M11))
print(np.shape(M12))
print(np.shape(M13))
print(np.shape(M21))
print(np.shape(M22))
print(np.shape(M23))
print(np.shape(M31))
print(np.shape(M32))
print(np.shape(M33))


print('=============== det of elementary matrices ===============')
print(la.det(M11.toarray()))
print(la.det(M12.toarray()))
print(la.det(M13.toarray()))
print(la.det(M21.toarray()))
print(la.det(M22.toarray()))
print(la.det(M23.toarray()))
print(la.det(M31.toarray()))
print(la.det(M32.toarray()))
print(la.det(M33.toarray()))


print('=============== cond of elementary matrices ===============')
print(la.cond(M11.toarray()))
print(la.cond(M12.toarray()))
print(la.cond(M13.toarray()))
print(la.cond(M21.toarray()))
print(la.cond(M22.toarray()))
print(la.cond(M23.toarray()))
print(la.cond(M31.toarray()))
print(la.cond(M32.toarray()))
print(la.cond(M33.toarray()))

print('=============== inv test for elementary matrices ===============')
try:
    la.inv(M11.toarray())
    print('M11: 1')
except:
    print('M11: 0')
    
try:
    la.inv(M12.toarray())
    print('M12: 1')
except:
    print('M12: 0')

try:
    la.inv(M13.toarray())
    print('M13: 1')
except:
    print('M13: 0')

try:
    la.inv(M21.toarray())
    print('M21: 1')
except:
    print('M21: 0')

try:
    la.inv(M22.toarray())
    print('M22: 1')
except:
    print('M12: 0')

try:
    la.inv(M23.toarray())
    print('M23: 1')
except:
    print('M23: 0')

try:
    la.inv(M31.toarray())
    print('M31: 1')
except:
    print('M31: 0')

try:
    la.inv(M32.toarray())
    print('M32: 1')
except:
    print('M32: 0')

try:
    la.inv(M33.toarray())
    print('M33: 1')
except:
    print('M33: 0')
    

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test1
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A1, A2, A3, B1, B2, B3, C1, C2, C3 = A11

A1_data = A1.data
A1_ind  = A1.indices
A1_ptr  = A1.indptr
A2_data = A2.data
A2_ind  = A2.indices
A2_ptr  = A2.indptr
A3_data = A3.data
A3_ind  = A3.indices
A3_ptr  = A3.indptr

B1_data = B1.data
B1_ind  = B1.indices
B1_ptr  = B1.indptr
B2_data = B2.data
B2_ind  = B2.indices
B2_ptr  = B2.indptr
B3_data = B3.data
B3_ind  = B3.indices
B3_ptr  = B3.indptr

C1_data = C1.data
C1_ind  = C1.indices
C1_ptr  = C1.indptr
C2_data = C2.data
C2_ind  = C2.indices
C2_ptr  = C2.indptr
C3_data = C3.data
C3_ind  = C3.indices
C3_ptr  = C3.indptr

alpha, beta, gamma = c11


N = np.shape(A1)[0]*np.shape(A2)[0]*np.shape(A3)[0]

b = np.ones(N)
x1 = np.zeros(N)#, dtype=float)
xref = la.solve(M11.toarray(), b)

kroneker_lower(A1_data, A1_ind, A1_ptr,
               A2_data, A2_ind, A2_ptr,
               A3_data, A3_ind, A3_ptr,
               B1_data, B1_ind, B1_ptr,
               B2_data, B2_ind, B2_ptr,
               B3_data, B3_ind, B3_ptr,
               C1_data, C1_ind, C1_ptr,
               C2_data, C2_ind, C2_ptr,
               C3_data, C3_ind, C3_ptr,
               alpha, beta, gamma,
               b, x1)

print(r'err_lower: %.e' %(max(abs(x1-xref))))

x1 = np.zeros(N)#, dtype=float)
kroneker_upper(A1_data, A1_ind, A1_ptr,
                A2_data, A2_ind, A2_ptr,
                A3_data, A3_ind, A3_ptr,
                B1_data, B1_ind, B1_ptr,
                B2_data, B2_ind, B2_ptr,
                B3_data, B3_ind, B3_ptr,
                C1_data, C1_ind, C1_ptr,
                C2_data, C2_ind, C2_ptr,
                C3_data, C3_ind, C3_ptr,
                alpha, beta, gamma,
                b, x1)


print(r'err_upper: %.e' %(max(abs(x1-xref))))


spsolve_kron_csr_3_sum_lower(A1_data, A1_ind, A1_ptr,
                           A2_data, A2_ind, A2_ptr,
                           A3_data, A3_ind, A3_ptr,
                           B1_data, B1_ind, B1_ptr,
                           B2_data, B2_ind, B2_ptr,
                           B3_data, B3_ind, B3_ptr,
                           C1_data, C1_ind, C1_ptr,
                           C2_data, C2_ind, C2_ptr,
                           C3_data, C3_ind, C3_ptr,
                           alpha, beta, gamma,
                           b, x1)

print(r'err_lower  old imp: %.e' %(max(abs(x1-xref))))

x1 = np.zeros(N)#, dtype=float)
spsolve_kron_csr_3_sum_upper(A1_data, A1_ind, A1_ptr,
                           A2_data, A2_ind, A2_ptr,
                           A3_data, A3_ind, A3_ptr,
                           B1_data, B1_ind, B1_ptr,
                           B2_data, B2_ind, B2_ptr,
                           B3_data, B3_ind, B3_ptr,
                           C1_data, C1_ind, C1_ptr,
                           C2_data, C2_ind, C2_ptr,
                           C3_data, C3_ind, C3_ptr,
                           alpha, beta, gamma,
                           b, x1)


print(r'err_upper  old imp: %.e' %(max(abs(x1-xref))))
