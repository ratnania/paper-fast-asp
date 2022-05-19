#!/usr/bin/env python
# coding: utf-8

# TODO NOT TESTED AFTER CREATING THIS FILE. ONLY THE LINEAR OPERATOR VERSION WAS
#      TESTED

import numpy as np
import scipy as sp

from numpy import linalg as la

from scipy.sparse import csr_matrix, coo_matrix, tril, triu
from scipy.sparse import bmat
from scipy.sparse import kron as sp_kron # TODO remove
from scipy.sparse import block_diag

from examples.asp_1d import M_matrix, K_matrix, D_matrix, R_matrix, H_matrix, I_matrix
from examples.asp_1d import knot_vector
from examples.asp_1d import Gauss_Legendre, non_zero_Curry_basis_functions, non_zero_basis_functions
from examples.asp_1d import location_matrix, location_matrix_2
from examples.asp_1d import basis_functions, Curry_basis_functions
from examples.asp_1d import Greville_points, collocation_matrix, histopolation_matrix

from multiprocess import Pool

# ========================================================================
def curl_matrix(T,p, tau, normalize, form):

    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    K1 = K_matrix(T1,p1)
    K2 = K_matrix(T2,p2)
    K3 = K_matrix(T3,p3)
    R1 = R_matrix(T1,p1, normalize=normalize)
    R2 = R_matrix(T2,p2, normalize=normalize)
    R3 = R_matrix(T3,p3, normalize=normalize)
    
    if form == 'csr':
        D1 = csr_matrix(D1)
        D2 = csr_matrix(D2)
        D3 = csr_matrix(D3)
        M1 = csr_matrix(M1)
        M2 = csr_matrix(M2)
        M3 = csr_matrix(M3)
        K1 = csr_matrix(K1)
        K2 = csr_matrix(K2)
        K3 = csr_matrix(K3)
        R1 = csr_matrix(R1)
        R2 = csr_matrix(R2)
        R3 = csr_matrix(R3)
        
    A11 = (D1, M2, K3, D1, K2, M3, D1, M2, M3)
    A12 = (R1, R2.T, M3)
    A13 = (R1, M2, R3.T)
    A21 = (R1.T, R2, M3.T)
    A22 = (K1, D2, M3, M1, D2, K3, M1, D2, M3)
    A23 = (M1, R2 , R3.T)
    A31 = (R1.T, M2, R3)
    A32 = (M1, R2.T, R3)
    A33 = (M1, K2, D3, K1, M2, D3, M1, M2, D3)
    
    c11 = (1., 1., tau)
    c12 = (-1., 0., 0.)
    c13 = (-1., 0., 0.)
    c21 = (-1., 0., 0.) 
    c22 = (1., 1., tau)
    c23 = (-1., 0., 0.)
    c31 = (-1., 0., 0.)
    c32 = (-1., 0., 0.)
    c33 = (1., 1., tau)
    
    
    
    matrices     = [A11, A12, A13, A21, A22, A23, A31, A32, A33]
    confficients = [c11, c12, c13, c21, c22, c23, c31, c32, c33]
    
    '''
    A11 = sp_kron(sp_kron(D1,M2),K3) + sp_kron(sp_kron(D1,K2),M3)
    A12 = -sp_kron(sp_kron(R1,R2.T),M3)
    A13 = -sp_kron(sp_kron(R1,M2),R3.T)
    A21 = -sp_kron(sp_kron(R1.T,R2),M3.T)
    A22 = sp_kron(sp_kron(K1,D2),M3)+sp_kron(sp_kron(M1,D2),K3)
    A23 = -sp_kron(sp_kron(M1,R2),R3.T)
    A31 = -sp_kron(sp_kron(R1.T,M2),R3)
    A32 = -sp_kron(sp_kron(M1,R2.T),R3)
    A33 = sp_kron(sp_kron(M1,K2),D3) + sp_kron(sp_kron(K1,M2),D3)
    
    Q1 = sp_kron(sp_kron(D1,M2),M3)
    Q2 = sp_kron(sp_kron(M1,D2),M3)
    Q3 = sp_kron(sp_kron(M1,M2),D3)


    A = bmat([[A11, A12, A13],
              [A21, A22, A23],
              [A31, A32, A33]])

    A = csr_matrix(A)
    A.eliminate_zeros()
    '''
    
    return matrices, confficients

# ========================================================================
def curl_shift_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)

    D1 = coo_matrix(D1)
    D2 = coo_matrix(D2)
    D3 = coo_matrix(D3)
    M1 = coo_matrix(M1)
    M2 = coo_matrix(M2)
    M3 = coo_matrix(M3)

    Q1 = sp_kron(sp_kron(D1,M2),M3)
    Q2 = sp_kron(sp_kron(M1,D2),M3)
    Q3 = sp_kron(sp_kron(M1,M2),D3)

    return block_diag([Q1, Q2, Q3])

# ========================================================================
def curl_shift_inv_matrix(T,p, normalize=True):
    
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    
    inv_D1 = la.inv(D1)
    inv_D2 = la.inv(D2)
    inv_D3 = la.inv(D3)
    inv_M1 = la.inv(M1)
    inv_M2 = la.inv(M2)
    inv_M3 = la.inv(M3)

    inv_D1 = csr_matrix(inv_D1)
    inv_D2 = csr_matrix(inv_D2)
    inv_D3 = csr_matrix(inv_D3)
    inv_M1 = csr_matrix(inv_M1)
    inv_M2 = csr_matrix(inv_M2)
    inv_M3 = csr_matrix(inv_M3)

    
    Q1 = sp_kron(sp_kron(inv_D1,inv_M2),inv_M3)
    Q2 = sp_kron(sp_kron(inv_M1,inv_D2),inv_M3)
    Q3 = sp_kron(sp_kron(inv_M1,inv_M2),inv_D3)

    return block_diag([Q1, Q2, Q3])

# ========================================================================

# ========================================================================
def curl_stiffness_matrix(T,p,tau, normalize=True):
    A1 = curl_matrix(T,p, normalize=normalize)
    A2 = curl_shift_matrix(T,p, normalize=normalize)
    A = A1 + tau*A2
    A.eliminate_zeros()
    return A

# ========================================================================
def curl_projection_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    I1 = I_matrix(T1,p1)
    I2 = I_matrix(T2,p2)
    I3 = I_matrix(T3,p3)
    H1 = H_matrix(T1,p1,normalize=normalize)
    H2 = H_matrix(T2,p2,normalize=normalize)
    H3 = H_matrix(T3,p3,normalize=normalize)

    I1 = coo_matrix(I1)
    I2 = coo_matrix(I2)
    I3 = coo_matrix(I3)
    H1 = coo_matrix(H1)
    H2 = coo_matrix(H2)
    H3 = coo_matrix(H3)

    P1 = sp_kron(sp_kron(H1,I2),I3)
    P2 = sp_kron(sp_kron(I1,H2),I3)
    P3 = sp_kron(sp_kron(I1,I2),H3)

    return block_diag([P1, P2, P3])

# ========================================================================
def grad_diagram_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])
    (m1,m2,m3) = (len(T1),len(T2),len(T3))
    (n1,n2,n3) = (m1-p1-1,m2-p2-1,m3-p3-1)

    I1 = np.eye(n1-2)
    I2 = np.eye(n2-2)
    I3 = np.eye(n3-2)

    D1 = np.diag(np.ones(n1-1),k=1)-np.diag(np.ones(n1))
    D2 = np.diag(np.ones(n2-1),k=1)-np.diag(np.ones(n2))
    D3 = np.diag(np.ones(n3-1),k=1)-np.diag(np.ones(n3))

    # ...
    if not normalize:
        raise NotImplementedError('')
    # ...

    D1 = D1[:-1,1:-1]
    D2 = D2[:-1,1:-1]
    D3 = D3[:-1,1:-1]

    I1 = coo_matrix(I1)
    I2 = coo_matrix(I2)
    I3 = coo_matrix(I3)
    D1 = coo_matrix(D1)
    D2 = coo_matrix(D2)
    D3 = coo_matrix(D3)

    G1 = sp_kron(sp_kron(D1,I2),I3)
    G2 = sp_kron(sp_kron(I1,D2),I3)
    G3 = sp_kron(sp_kron(I1,I2),D3)

    G = bmat([[G1],[G2],[G3]])

    return G

# ========================================================================
def curl_asp_matrix_1(T,p,tau,PCurl):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    K1 = K_matrix(T1,p1)
    K2 = K_matrix(T2,p2)
    K3 = K_matrix(T3,p3)

    # convert to csr
    M1 = csr_matrix(M1)
    M2 = csr_matrix(M2)
    M3 = csr_matrix(M3)
    K1 = csr_matrix(K1)
    K2 = csr_matrix(K2)
    K3 = csr_matrix(K3)

    # ...
    B = sp_kron(sp_kron(M1,M2),M3)

    L1 = sp_kron(sp_kron(K1,M2),M3)
    L2 = sp_kron(sp_kron(M1,K2),M3)
    L3 = sp_kron(sp_kron(M1,M2),K3)
    L  = L1+L2+L3

    LB = L+tau*B
    # ...

    # ...
    inv_LB  = csr_matrix(np.linalg.inv(LB.toarray()))
    inv_L   = block_diag([inv_LB, inv_LB, inv_LB])
    t_PCurl = PCurl.transpose().tocsr()
    # ...

    return PCurl * inv_L * t_PCurl

# ========================================================================
def curl_asp_matrix_2(T,p,tau,G):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    K1 = K_matrix(T1,p1)
    K2 = K_matrix(T2,p2)
    K3 = K_matrix(T3,p3)

    # convert to csr
    M1 = csr_matrix(M1)
    M2 = csr_matrix(M2)
    M3 = csr_matrix(M3)
    K1 = csr_matrix(K1)
    K2 = csr_matrix(K2)
    K3 = csr_matrix(K3)

    # ...
    L1 = sp_kron(sp_kron(K1,M2),M3)
    L2 = sp_kron(sp_kron(M1,K2),M3)
    L3 = sp_kron(sp_kron(M1,M2),K3)
    L = L1+L2+L3
    # ...

    # ...
    inv_L = csr_matrix(np.linalg.inv(L.toarray()))
    t_G = G.transpose().tocsr()
    # ...

    return 1./tau * G * inv_L * t_G

# ========================================================================
def curl_create_matrices(N, p, tau, normalize=True, verbose=False):

    T1=knot_vector(N[0],p[0])
    T2=knot_vector(N[1],p[1])
    T3=knot_vector(N[2],p[2])
    T=(T1,T2,T3)

    # ...
    A = curl_stiffness_matrix(T,p,tau, normalize=normalize)
    if verbose:
        print('> assemble A: done')
    # ...

    # ...
    PCurl = curl_projection_matrix(T,p,normalize=normalize)
    if verbose:
        print('> assemble PCurl: done')
    # ...

    # ...
    G = grad_diagram_matrix(T,p, normalize=normalize)
    if verbose:
        print('> assemble G: done')
    # ...

    # ... convert to sparse matrices
    A = csr_matrix(A)
    PCurl = csr_matrix(PCurl)
    G = csr_matrix(G)
    # ...

    # ...
    B = curl_asp_matrix_1(T,p,tau,PCurl) + curl_asp_matrix_2(T,p,tau,G)
    B = csr_matrix(B)
    # ...

    return A, B

# ========================================================================
def curl_GLT_matrix(N, p, normalize=True):
    
    T1 = knot_vector(N[0],p[0])
    T2 = knot_vector(N[1],p[1])
    T3 = knot_vector(N[2],p[2])
    T  = (T1,T2,T3)
    
    M  = curl_shift_inv_matrix(T,p, normalize=normalize)
    
    return M

'''
# ========================================================================
def curl_rhs_vector(N, p, f, normalize=True, separated_variables = True):
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0], p1)
    T2           = knot_vector(N[1], p2)
    T3           = knot_vector(N[2], p3)
    
    if separated_variables == True:
        print('TODO A SIMPLE IMPLEMENTATION IN THE CASE WHERE A FUNCTION WITH SEPARATED VARIABLES IS GIVEN')
    
    else:
        (N1, N2, N3)                   = N
        (m1, m2, m3)                   = (len(T1), len(T2), len(T3))
        (n1, n2, n3)                   = (m1-p1-1, m2-p2-1, m3-p3-1)
        (ordergl1, ordergl2, ordergl3) = (p1+1, p2+1, p3+1)
        (r1, w1)                       = Gauss_Legendre(ordergl = ordergl1,
                                                        tol     = 10e-14) 
        (r2, w2)                       = Gauss_Legendre(ordergl = ordergl2,
                                                        tol     = 10e-14)
        (r3, w3)                       = Gauss_Legendre(ordergl = ordergl3,
                                                        tol     = 10e-14) 
        (k1, k2, k3)                   = (len(r1), len(r2), len(r3))
        
        rhs1                           = np.zeros((n1-1, n2, n3), dtype = float)
        rhs2                           = np.zeros((n1, n2-1, n3), dtype = float)
        rhs3                           = np.zeros((n1, n2, n3-1), dtype = float)
        
        for e1 in range(N1):
            a1 = T1[p1+e1]
            b1 = T1[p1+e1+1]
            h1 = b1-a1
            
            i1_l = location_matrix_2(T=T1, p=p1)[e1,:][0]
            i1_r = location_matrix_2(T=T1, p=p1)[e1,:][-1]+1
            j1_l = location_matrix(T=T1, p=p1)[e1,:][0]
            j1_r = location_matrix(T=T1, p=p1)[e1,:][-1]+1
            k1_l = location_matrix(T=T1, p=p1)[e1,:][0]
            k1_r = location_matrix(T=T1, p=p1)[e1,:][-1]+1
            for e2 in range(N2):
                a2   = T2[p2+e2]
                b2   = T2[p2+e2+1]
                h2   = b2-a2
                
                i2_l = location_matrix(T=T2, p=p2)[e2,:][0]
                i2_r = location_matrix(T=T2, p=p2)[e2,:][-1]+1
                j2_l = location_matrix_2(T=T2, p=p2)[e2,:][0]
                j2_r = location_matrix_2(T=T2, p=p2)[e2,:][-1]+1
                k2_l = location_matrix(T=T2, p=p2)[e2,:][0]
                k2_r = location_matrix(T=T2, p=p2)[e2,:][-1]+1
                for e3 in range(N3):
                    a3   = T3[p3+e3]
                    b3   = T3[p3+e3+1]
                    h3   = b3-a3
                    
                    i3_l = location_matrix(T=T3, p=p3)[e3,:][0]
                    i3_r = location_matrix(T=T3, p=p3)[e3,:][-1]+1
                    j3_l = location_matrix(T=T3, p=p3)[e3,:][0]
                    j3_r = location_matrix(T=T3, p=p3)[e3,:][-1]+1
                    k3_l = location_matrix_2(T=T3, p=p3)[e3,:][0]
                    k3_r = location_matrix_2(T=T3, p=p3)[e3,:][-1]+1
                    
                    vol1 = np.zeros(p1*(p2+1)*(p3+1), dtype = float)
                    vol2 = np.zeros((p1+1)*p2*(p3+1), dtype = float)
                    vol3 = np.zeros((p1+1)*(p2+1)*p3, dtype = float)
                    
                    vol1 = csr_matrix(vol1)
                    vol2 = csr_matrix(vol2)
                    vol3 = csr_matrix(vol3)
                
                    for s1 in range(k1):
                        t1 = 0.5*(h1*r1[s1]+a1+b1)
                        for s2 in range(k2):
                            t2 = 0.5*(h2*r2[s2]+a2+b2)
                            for s3 in range(k3):
                                t3 = 0.5*(h3*r3[s3]+a3+b3)
                                
                                B1 = non_zero_basis_functions(T = T1, p = p1, t = t1)
                                B2 = non_zero_basis_functions(T = T2, p = p2, t = t2)
                                B3 = non_zero_basis_functions(T = T3, p = p3, t = t3)
                                D1 = non_zero_Curry_basis_functions(T = T1, p = p1, t = t1, normalize = normalize)
                                D2 = non_zero_Curry_basis_functions(T = T2, p = p2, t = t2, normalize = normalize)
                                D3 = non_zero_Curry_basis_functions(T = T3, p = p3, t = t3, normalize = normalize)
                                
                                B1 = csr_matrix(B1)
                                B2 = csr_matrix(B2)
                                B3 = csr_matrix(B3)
                                D1 = csr_matrix(D1)
                                D2 = csr_matrix(D2)
                                D3 = csr_matrix(D3)
                            
                                vol1 += sp_kron(sp_kron(D1,B2),B3)*f(t1, t2, t3)[0]*w1[s1]*w2[s2]*w3[s3]
                                vol2 += sp_kron(sp_kron(B1,D2),B3)*f(t1, t2, t3)[1]*w1[s1]*w2[s2]*w3[s3]
                                vol3 += sp_kron(sp_kron(B1,B2),D3)*f(t1, t2, t3)[2]*w1[s1]*w2[s2]*w3[s3]
                           
                    vol1 = 0.125*h1*h2*h3*vol1 
                    vol2 = 0.125*h1*h2*h3*vol2 
                    vol3 = 0.125*h1*h2*h3*vol3 
                    
                    vol1 = vol1.toarray()
                    vol2 = vol2.toarray()
                    vol3 = vol3.toarray()
                    
                    vol1 = vol1.reshape((p1, p2+1, p3+1))
                    vol2 = vol2.reshape((p1+1, p2, p3+1))
                    vol3 = vol3.reshape((p1+1, p2+1, p3))
                    
                    rhs1[i1_l:i1_r, i2_l:i2_r, i3_l:i3_r] += vol1
                    rhs2[j1_l:j1_r, j2_l:j2_r, j3_l:j3_r] += vol2
                    rhs3[k1_l:k1_r, k2_l:k2_r, k3_l:k3_r] += vol3
                    
        rhs1 = rhs1[:, 1:-1, 1:-1] 
        rhs2 = rhs2[1:-1, :, 1:-1]
        rhs3 = rhs3[1:-1, 1:-1, :]
    
        rhs1 = rhs1.flatten()
        rhs2 = rhs2.flatten()
        rhs3 = rhs3.flatten()
    
        rhs = np.hstack((np.hstack((rhs1, rhs2)), rhs3))
        
    return rhs         
'''                 

def curl_rhs_elementary_vector(e1, e2, e3, N, p, f, normalize=True, separated_variables = True):
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0], p1)
    T2           = knot_vector(N[1], p2)
    T3           = knot_vector(N[2], p3)

    (ordergl1, ordergl2, ordergl3) = (p1+1, p2+1, p3+1)
    (r1, w1)                       = Gauss_Legendre(ordergl = ordergl1,
                                                    tol     = 10e-14) 
    (r2, w2)                       = Gauss_Legendre(ordergl = ordergl2,
                                                    tol     = 10e-14)
    (r3, w3)                       = Gauss_Legendre(ordergl = ordergl3,
                                                    tol     = 10e-14) 
    (k1, k2, k3)                   = (len(r1), len(r2), len(r3))
    
    
    
    a1 = T1[p1+e1]
    b1 = T1[p1+e1+1]
    h1 = b1-a1
    
    
    a2   = T2[p2+e2]
    b2   = T2[p2+e2+1]
    h2   = b2-a2
    
    a3   = T3[p3+e3]
    b3   = T3[p3+e3+1]
    h3   = b3-a3
    
    
    vol1 = np.zeros(p1*(p2+1)*(p3+1), dtype = float)
    vol2 = np.zeros((p1+1)*p2*(p3+1), dtype = float)
    vol3 = np.zeros((p1+1)*(p2+1)*p3, dtype = float)
    
    vol1 = csr_matrix(vol1)
    vol2 = csr_matrix(vol2)
    vol3 = csr_matrix(vol3)
    
    for s1 in range(k1):
        t1 = 0.5*(h1*r1[s1]+a1+b1)
        for s2 in range(k2):
            t2 = 0.5*(h2*r2[s2]+a2+b2)
            for s3 in range(k3):
                t3 = 0.5*(h3*r3[s3]+a3+b3)
                
                B1 = non_zero_basis_functions(T = T1, p = p1, t = t1)
                B2 = non_zero_basis_functions(T = T2, p = p2, t = t2)
                B3 = non_zero_basis_functions(T = T3, p = p3, t = t3)
                D1 = non_zero_Curry_basis_functions(T = T1, p = p1, t = t1, normalize = normalize)
                D2 = non_zero_Curry_basis_functions(T = T2, p = p2, t = t2, normalize = normalize)
                D3 = non_zero_Curry_basis_functions(T = T3, p = p3, t = t3, normalize = normalize)
                
                B1 = csr_matrix(B1)
                B2 = csr_matrix(B2)
                B3 = csr_matrix(B3)
                D1 = csr_matrix(D1)
                D2 = csr_matrix(D2)
                D3 = csr_matrix(D3)
            
                vol1 += sp_kron(sp_kron(D1,B2),B3)*f(t1, t2, t3)[0]*w1[s1]*w2[s2]*w3[s3]
                vol2 += sp_kron(sp_kron(B1,D2),B3)*f(t1, t2, t3)[1]*w1[s1]*w2[s2]*w3[s3]
                vol3 += sp_kron(sp_kron(B1,B2),D3)*f(t1, t2, t3)[2]*w1[s1]*w2[s2]*w3[s3]
           
    vol1 = 0.125*h1*h2*h3*vol1 
    vol2 = 0.125*h1*h2*h3*vol2 
    vol3 = 0.125*h1*h2*h3*vol3 
    
    vol1 = vol1.toarray()
    vol2 = vol2.toarray()
    vol3 = vol3.toarray()
    
    vol1 = vol1.reshape((p1, p2+1, p3+1))
    vol2 = vol2.reshape((p1+1, p2, p3+1))
    vol3 = vol3.reshape((p1+1, p2+1, p3))
    
    return [vol1, vol2, vol3]

    
# ========================================================================
def curl_elementary_vectors(N, p, f, normalize=True,  separated_variables = False):
    
    (N1, N2, N3) = N
    
    global elementary_vector_function
    
    def elementary_vector_function(e1, e2, e3) : 
        return curl_rhs_elementary_vector(e1, e2, e3, N, p, f, 
                                          normalize           = normalize, 
                                          separated_variables = separated_variables)  
    
    
    
    process_pool = Pool(3)
    data = [(e1, e2, e3) for e1 in range(N1) for e2 in range(N2) for e3 in range(N3)]
    elementary_vectors = process_pool.starmap(elementary_vector_function , data)
    
    
    return elementary_vectors


# ========================================================================
def curl_rhs_vector(N, p, f, normalize=True, separated_variables = True):
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0], p1)
    T2           = knot_vector(N[1], p2)
    T3           = knot_vector(N[2], p3)
    
    if separated_variables == True:
        print('TODO A SIMPLE IMPLEMENTATION IN THE CASE WHERE A FUNCTION WITH SEPARATED VARIABLES IS GIVEN')
    
    else:
        (N1, N2, N3)                   = N
        (m1, m2, m3)                   = (len(T1), len(T2), len(T3))
        (n1, n2, n3)                   = (m1-p1-1, m2-p2-1, m3-p3-1)
        
        rhs1                           = np.zeros((n1-1, n2, n3), dtype = float)
        rhs2                           = np.zeros((n1, n2-1, n3), dtype = float)
        rhs3                           = np.zeros((n1, n2, n3-1), dtype = float)
        
        elementary_vectors             = curl_elementary_vectors(N, p, f, 
                                                                 normalize           = normalize,  
                                                                 separated_variables = separated_variables)
        
        
        for e1 in range(N1):
            i1_l = location_matrix_2(T=T1, p=p1)[e1,:][0]
            i1_r = location_matrix_2(T=T1, p=p1)[e1,:][-1]+1
            j1_l = location_matrix(T=T1, p=p1)[e1,:][0]
            j1_r = location_matrix(T=T1, p=p1)[e1,:][-1]+1
            k1_l = location_matrix(T=T1, p=p1)[e1,:][0]
            k1_r = location_matrix(T=T1, p=p1)[e1,:][-1]+1
            
            for e2 in range(N2):
                
                i2_l = location_matrix(T=T2, p=p2)[e2,:][0]
                i2_r = location_matrix(T=T2, p=p2)[e2,:][-1]+1
                j2_l = location_matrix_2(T=T2, p=p2)[e2,:][0]
                j2_r = location_matrix_2(T=T2, p=p2)[e2,:][-1]+1
                k2_l = location_matrix(T=T2, p=p2)[e2,:][0]
                k2_r = location_matrix(T=T2, p=p2)[e2,:][-1]+1
                
                for e3 in range(N3):
                    i3_l = location_matrix(T=T3, p=p3)[e3,:][0]
                    i3_r = location_matrix(T=T3, p=p3)[e3,:][-1]+1
                    j3_l = location_matrix(T=T3, p=p3)[e3,:][0]
                    j3_r = location_matrix(T=T3, p=p3)[e3,:][-1]+1
                    k3_l = location_matrix_2(T=T3, p=p3)[e3,:][0]
                    k3_r = location_matrix_2(T=T3, p=p3)[e3,:][-1]+1
                    
                    [vol1, vol2, vol3] = elementary_vectors[e3+e2*N3+e1*N2*N3]
                    
                    rhs1[i1_l:i1_r, i2_l:i2_r, i3_l:i3_r] += vol1
                    rhs2[j1_l:j1_r, j2_l:j2_r, j3_l:j3_r] += vol2
                    rhs3[k1_l:k1_r, k2_l:k2_r, k3_l:k3_r] += vol3
        
        rhs1 = rhs1[:, 1:-1, 1:-1] 
        rhs2 = rhs2[1:-1, :, 1:-1]
        rhs3 = rhs3[1:-1, 1:-1, :]            
                    
        rhs1 = rhs1.flatten()
        rhs2 = rhs2.flatten()
        rhs3 = rhs3.flatten()
    
        rhs = np.hstack((np.hstack((rhs1, rhs2)), rhs3))
        
    return rhs                     

# ========================================================================
def curl_appr_sol_function(N, p, u, x, normalize=True):
    
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0],p1)
    T2           = knot_vector(N[1],p2)
    T3           = knot_vector(N[2],p3)
    
    if len(u)%3 == 0:
        u1 = u[:len(u)//3]
        u2 = u[len(u)//3:2*(len(u)//3)]
        u3 = u[2*(len(u)//3):]
    else:
        raise NotImplementedError('')
        
    (x1, x2, x3) = x
    
    b1 = basis_functions(T1, p1, x1)[1:-1]
    b2 = basis_functions(T2, p2, x2)[1:-1]
    b3 = basis_functions(T3, p3, x3)[1:-1]
    d1 = Curry_basis_functions(T1,p1,x1, normalize=normalize)
    d2 = Curry_basis_functions(T2,p2,x2, normalize=normalize)
    d3 = Curry_basis_functions(T3,p3,x3, normalize=normalize)
    
    c1 = sp_kron(sp_kron(d1,b2),b3)
    c2 = sp_kron(sp_kron(b1,d2),b3) 
    c3 = sp_kron(sp_kron(b1,b2),d3)
    
    c1 = np.squeeze(c1.toarray())
    c2 = np.squeeze(c2.toarray())
    c3 = np.squeeze(c3.toarray())

    uh1 = np.dot(u1, c1)
    uh2 = np.dot(u2, c2)
    uh3 = np.dot(u3, c3)
    
    uh = (uh1, uh2, uh3)
    
    return uh            
    
def curl_DeRham_projection_coefficients_vector(N, p, u, normalize=True, separated_variables = False):
    
    (p1,p2,p3) = (p[0],p[1],p[2])
    T1         = knot_vector(N[0],p1)
    T2         = knot_vector(N[1],p2)
    T3         = knot_vector(N[2],p3)
    
    if separated_variables == True:
        print('TODO A SIMPLE IMPLEMENTATION IN THE CASE WHERE A FUNCTION WITH SEPARATED VARIABLES IS GIVEN')
    
    else:
        (N1, N2, N3)             = N
        (m1, m2, m3)             = (len(T1), len(T2), len(T3))
        (n1, n2, n3)             = (m1-p1-1, m2-p2-1, m3-p3-1)
        (ordergl1, ordergl2, ordergl3) = (p1+1, p2+1, p3+1)
        (r1, w1)             = Gauss_Legendre(ordergl = ordergl1,
                                              tol     = 10e-14) 
        (r2, w2)             = Gauss_Legendre(ordergl = ordergl2,
                                              tol     = 10e-14) 
        (r3, w3)             = Gauss_Legendre(ordergl = ordergl3,
                                              tol     = 10e-14) 
        (k1, k2, k3)             = (len(r1), len(r2), len(r3))
        
        g1                   = Greville_points(T=T1, p=p1)
        g2                   = Greville_points(T=T2, p=p2)
        g3                   = Greville_points(T=T3, p=p3)
    
        y1                   = np.zeros((n1-1, n2, n3), dtype = float)
        y2                   = np.zeros((n1, n2-1, n3), dtype = float)
        y3                   = np.zeros((n1, n2, n3-1), dtype = float)
        
        for i1 in range(n1-1):
            a = g1[i1]
            b = g1[i1+1]
            h = b-a
            for i2 in range(n2):
                for i3 in range(n3):    
                    vol = 0.0
                    for s in range(k1):
                        t = 0.5*(h*r1[s]+a+b)
                        vol += u(t, g2[i2], g3[i3])[0]*w1[s]
                    y1[i1][i2][i3] = 0.5*h*vol
        
        for i1 in range(n1):
            for i2 in range(n2-1):
                a   = g2[i2]
                b   = g2[i2+1]
                h   = b-a
                for i3 in range(n3): 
                    vol = 0.0
                    for s in range(k2):
                        t = 0.5*(h*r2[s]+a+b)
                        vol += u(g1[i1], t, g1[i3])[1]*w2[s]
                    y2[i1][i2][i3] = 0.5*h*vol 
        
        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3-1): 
                    a   = g3[i3]
                    b   = g3[i3+1]
                    h   = b-a
                    vol = 0.0
                    for s in range(k3):
                        t = 0.5*(h*r3[s]+a+b)
                        vol += u(g1[i1], g2[i2], t)[2]*w3[s]
                    y3[i1][i2][i3] = 0.5*h*vol 
                    
        y1 = y1[:, 1:-1, 1:-1].flatten()
        y2 = y2[1:-1, :, 1:-1].flatten()
        y3 = y3[1:-1, 1:-1, :].flatten()
        
        M1 = collocation_matrix(T1, p1)[1:-1, 1:-1]
        M2 = collocation_matrix(T2, p2)[1:-1, 1:-1]
        M3 = collocation_matrix(T3, p3)[1:-1, 1:-1]
        
        H1 = histopolation_matrix(T1, p1, normalize=normalize)
        H2 = histopolation_matrix(T2, p2, normalize=normalize)
        H3 = histopolation_matrix(T3, p3, normalize=normalize)
        
        P1 = sp_kron(sp_kron(H1, M2), M3)
        P2 = sp_kron(sp_kron(M1, H2), M3)
        P3 = sp_kron(sp_kron(M1, M2), H3)
        
        P1 = P1.toarray()
        P2 = P2.toarray()
        P3 = P3.toarray()
    
        c1 = la.solve(P1, y1)
        c2 = la.solve(P2, y2)
        c3 = la.solve(P3, y3)
        
        c  = np.hstack((c1, c2, c3))
        return c
    


    
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#               Matrices/vectors reltated to div porblem
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ========================================================================
def div_matrix(T,p, normalize=True):

    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])
    
    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    K1 = K_matrix(T1,p1)
    K2 = K_matrix(T2,p2)
    K3 = K_matrix(T3,p3)
    R1 = R_matrix(T1,p1, normalize=normalize)
    R2 = R_matrix(T2,p2, normalize=normalize)
    R3 = R_matrix(T3,p3, normalize=normalize)

    D1 = coo_matrix(D1)
    D2 = coo_matrix(D2)
    D3 = coo_matrix(D3)
    K1 = coo_matrix(K1)
    K2 = coo_matrix(K2)
    K3 = coo_matrix(K3)
    R1 = coo_matrix(R1)
    R2 = coo_matrix(R2)
    R3 = coo_matrix(R3)

    A11 = sp_kron(sp_kron(K1,D2),D3) 
    A12 = sp_kron(sp_kron(R1.T,R2),D3)
    A13 = sp_kron(sp_kron(R1.T,D2),R3)
    A21 = sp_kron(sp_kron(R1,R2.T),D3)
    A22 = sp_kron(sp_kron(D1,K2),D3)
    A23 = sp_kron(sp_kron(D1,R2.T),R3)
    A31 = sp_kron(sp_kron(R1,D2),R3.T)
    A32 = sp_kron(sp_kron(D1,R2),R3.T)
    A33 = sp_kron(sp_kron(D1,D2),K3) 

    A11.eliminate_zeros()
    A12.eliminate_zeros()
    A13.eliminate_zeros()
    A21.eliminate_zeros()
    A22.eliminate_zeros()
    A23.eliminate_zeros()
    A31.eliminate_zeros()
    A32.eliminate_zeros()
    A33.eliminate_zeros()

    A = bmat([[A11, A12, A13],
              [A21, A22, A23],
              [A31, A32, A33]])

    A = csr_matrix(A)
    A.eliminate_zeros()
    return A

# ========================================================================
def div_shift_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)

    D1 = coo_matrix(D1)
    D2 = coo_matrix(D2)
    D3 = coo_matrix(D3)
    M1 = coo_matrix(M1)
    M2 = coo_matrix(M2)
    M3 = coo_matrix(M3)

    Q1 = sp_kron(sp_kron(M1,D2),D3)
    Q2 = sp_kron(sp_kron(D1,M2),D3)
    Q3 = sp_kron(sp_kron(D1,D2),M3)

    return block_diag([Q1, Q2, Q3])

# ========================================================================
def div_shift_inv_matrix(T,p, normalize=True):
    
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    D1 = D_matrix(T1,p1, normalize=normalize)
    D2 = D_matrix(T2,p2, normalize=normalize)
    D3 = D_matrix(T3,p3, normalize=normalize)
    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    
    inv_D1 = la.inv(D1)
    inv_D2 = la.inv(D2)
    inv_D3 = la.inv(D3)
    inv_M1 = la.inv(M1)
    inv_M2 = la.inv(M2)
    inv_M3 = la.inv(M3)

    inv_D1 = csr_matrix(inv_D1)
    inv_D2 = csr_matrix(inv_D2)
    inv_D3 = csr_matrix(inv_D3)
    inv_M1 = csr_matrix(inv_M1)
    inv_M2 = csr_matrix(inv_M2)
    inv_M3 = csr_matrix(inv_M3)

    
    Q1 = sp_kron(sp_kron(inv_M1,inv_D2),inv_D3)
    Q2 = sp_kron(sp_kron(inv_D1,inv_M2),inv_D3)
    Q3 = sp_kron(sp_kron(inv_D1,inv_D2),inv_M3)

    return block_diag([Q1, Q2, Q3])

# ========================================================================
def div_Stiffness_Matrix(T,p,tau, normalize=True):
    A1 = div_matrix(T,p, normalize=normalize)
    A2 = div_shift_matrix(T,p, normalize=normalize)
    A = A1 + tau*A2
    A.eliminate_zeros()
    return A

# ========================================================================
def div_projection_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    I1 = I_matrix(T1,p1)
    I2 = I_matrix(T2,p2)
    I3 = I_matrix(T3,p3)
    H1 = H_matrix(T1,p1,normalize=normalize)
    H2 = H_matrix(T2,p2,normalize=normalize)
    H3 = H_matrix(T3,p3,normalize=normalize)

    I1 = coo_matrix(I1)
    I2 = coo_matrix(I2)
    I3 = coo_matrix(I3)
    H1 = coo_matrix(H1)
    H2 = coo_matrix(H2)
    H3 = coo_matrix(H3)

    P1 = sp_kron(sp_kron(I1,H2),H3)
    P2 = sp_kron(sp_kron(H1,I2),H3)
    P3 = sp_kron(sp_kron(H1,H2),I3)

    return block_diag([P1, P2, P3])

# ========================================================================
def curl_diagram_matrix(T,p, normalize=True):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])
    (m1,m2,m3) = (len(T1),len(T2),len(T3))
    (n1,n2,n3) = (m1-p1-1,m2-p2-1,m3-p3-1)

    I1 = np.eye(n1-2)
    I2 = np.eye(n2-2)
    I3 = np.eye(n3-2)
    
    I1_ = np.eye(n1-1)
    I2_ = np.eye(n2-1)
    I3_ = np.eye(n3-1)

    D1 = np.diag(np.ones(n1-1),k=1)-np.diag(np.ones(n1))
    D2 = np.diag(np.ones(n2-1),k=1)-np.diag(np.ones(n2))
    D3 = np.diag(np.ones(n3-1),k=1)-np.diag(np.ones(n3))

    # ...
    if not normalize:
        raise NotImplementedError('')
    # ...

    D1 = D1[:-1,1:-1]
    D2 = D2[:-1,1:-1]
    D3 = D3[:-1,1:-1]

    I1 = coo_matrix(I1)
    I2 = coo_matrix(I2)
    I3 = coo_matrix(I3)
    I1_ = coo_matrix(I1_)
    I2_ = coo_matrix(I2_)
    I3_ = coo_matrix(I3_)
    D1 = coo_matrix(D1)
    D2 = coo_matrix(D2)
    D3 = coo_matrix(D3)

    G12 = -sp_kron(sp_kron(I1,I2_),D3)
    G13 = sp_kron(sp_kron(I1,D2),I3_)
    G21 = sp_kron(sp_kron(I1_,I2),D3)
    G23 = -sp_kron(sp_kron(D1,I2),I3_)
    G31 = -sp_kron(sp_kron(I1_,D2),I3)
    G32 = sp_kron(sp_kron(D1,I2_),I3)
    
    G11 = np.zeros((np.shape(G12)[0],np.shape(G21)[1]))
    G22 = np.zeros((np.shape(G21)[0],np.shape(G32)[1]))
    G33 = np.zeros((np.shape(G32)[0],np.shape(G23)[1]))
    
    G11 = coo_matrix(G11)
    G22 = coo_matrix(G22)
    G33 = coo_matrix(G33)
    

    G = bmat([[G11, G12, G13],
              [G21, G22, G23],
              [G31, G32, G33]])

    return  G

# ========================================================================
def div_matrix_B1(T,p,tau,Pdiv):
    (p1,p2,p3) = (p[0],p[1],p[2])
    (T1,T2,T3) = (T[0],T[1],T[2])

    M1 = M_matrix(T1,p1)
    M2 = M_matrix(T2,p2)
    M3 = M_matrix(T3,p3)
    K1 = K_matrix(T1,p1)
    K2 = K_matrix(T2,p2)
    K3 = K_matrix(T3,p3)

    # convert to csr
    M1 = csr_matrix(M1)
    M2 = csr_matrix(M2)
    M3 = csr_matrix(M3)
    K1 = csr_matrix(K1)
    K2 = csr_matrix(K2)
    K3 = csr_matrix(K3)

    # ...
    B = sp_kron(sp_kron(M1,M2),M3)

    L1 = sp_kron(sp_kron(K1,M2),M3)
    L2 = sp_kron(sp_kron(M1,K2),M3)
    L3 = sp_kron(sp_kron(M1,M2),K3)
    L = L1+L2+L3

    LB = L+tau*B
    # ...
    
    # ...
    inv_LB = csr_matrix(np.linalg.inv(LB.toarray()))
    inv_L = block_diag([inv_LB, inv_LB, inv_LB])
    t_Pdiv = Pdiv.transpose().tocsr()
    # ...

    return Pdiv * inv_L * t_Pdiv

# ========================================================================
def div_matrix_B2(T,p,tau, curl_smother = 'Gauss_Seidel'):
    
    ACurl = curl_matrix(T,p, normalize=True)

    if curl_smother == 'Jacobi':
        R = sp.sparse.diags(1./ACurl.diagonal())
    elif curl_smother == 'Gauss_Seidel':
        L_Curl = tril(ACurl,    format="csr")
        U_Curl = triu(ACurl,  format="csr")
        
        L_Curl_inv = la.inv(L_Curl.toarray())
        U_Curl_inv = la.inv(U_Curl.toarray())
         
        L_Curl_inv = csr_matrix(L_Curl_inv)
        #++++++++++++++++++++++++++++++++++
        U_Curl_inv = csr_matrix(U_Curl_inv)
        
        R = L_Curl_inv + U_Curl_inv - (U_Curl_inv @ ACurl) @ L_Curl_inv
        
    C = curl_diagram_matrix(T,p, normalize=True)
    
    # ...
    t_C = C.transpose().tocsr()
    # ...

    return 1./tau * C * R * t_C

# ========================================================================
def div_create_matrices(N, p, tau, normalize=True, verbose=False, curl_smother = 'Gauss_Seidel'):

    T1=knot_vector(N[0],p[0])
    T2=knot_vector(N[1],p[1])
    T3=knot_vector(N[2],p[2])
    T=(T1,T2,T3)

    # ...
    A = div_Stiffness_Matrix(T,p,tau, normalize=normalize)
    if verbose:
        print('> assemble A: done')
    # ...

    # ...
    PCurl = curl_projection_matrix(T,p,normalize=normalize)
    if verbose:
        print('> assemble PCurl: done')
    # ...
    
    # ...
    Pdiv = div_projection_matrix(T,p,normalize=normalize)
    if verbose:
        print('> assemble Pdiv: done')
    # ...
    
    # ...
    C = curl_diagram_matrix(T,p,normalize=normalize)
    if verbose:
        print('> assemble C: done')
    # ...

    # ... convert to sparse matrices
    A = csr_matrix(A)
    PCurl = csr_matrix(PCurl)
    Pdiv = csr_matrix(Pdiv)
    C = csr_matrix(C)
    CPCurl = C*PCurl
    # ...

    # ...
    B = div_matrix_B1(T,p,tau,Pdiv) + div_matrix_B1(T,p,tau,CPCurl) + div_matrix_B2(T,p,tau, curl_smother)
    # B = csr_matrix(B)
    # ...

    return A, B


# ========================================================================
def div_GLT_matrix(N, p, normalize=True):
    
    T1 = knot_vector(N[0],p[0])
    T2 = knot_vector(N[1],p[1])
    T3 = knot_vector(N[2],p[2])
    T  = (T1,T2,T3)
    
    M  = div_shift_inv_matrix(T,p, normalize=normalize)
    
    return M
    
def div_rhs_elementary_vector(e1, e2, e3, N, p, f, normalize=True, separated_variables = True):
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0], p1)
    T2           = knot_vector(N[1], p2)
    T3           = knot_vector(N[2], p3)

    (ordergl1, ordergl2, ordergl3) = (p1+1, p2+1, p3+1)
    (r1, w1)                       = Gauss_Legendre(ordergl = ordergl1,
                                                    tol     = 10e-14) 
    (r2, w2)                       = Gauss_Legendre(ordergl = ordergl2,
                                                    tol     = 10e-14)
    (r3, w3)                       = Gauss_Legendre(ordergl = ordergl3,
                                                    tol     = 10e-14) 
    (k1, k2, k3)                   = (len(r1), len(r2), len(r3))
    
    
    
    a1 = T1[p1+e1]
    b1 = T1[p1+e1+1]
    h1 = b1-a1
    
    
    a2   = T2[p2+e2]
    b2   = T2[p2+e2+1]
    h2   = b2-a2
    
    a3   = T3[p3+e3]
    b3   = T3[p3+e3+1]
    h3   = b3-a3
    
    
    vol1 = np.zeros((p1+1)*p2*p3, dtype = float)
    vol2 = np.zeros(p1*(p2+1)*p3, dtype = float)
    vol3 = np.zeros(p1*p2*(p3+1), dtype = float)
    
    vol1 = csr_matrix(vol1)
    vol2 = csr_matrix(vol2)
    vol3 = csr_matrix(vol3)
    
    for s1 in range(k1):
        t1 = 0.5*(h1*r1[s1]+a1+b1)
        for s2 in range(k2):
            t2 = 0.5*(h2*r2[s2]+a2+b2)
            for s3 in range(k3):
                t3 = 0.5*(h3*r3[s3]+a3+b3)
                
                B1 = non_zero_basis_functions(T = T1, p = p1, t = t1)
                B2 = non_zero_basis_functions(T = T2, p = p2, t = t2)
                B3 = non_zero_basis_functions(T = T3, p = p3, t = t3)
                D1 = non_zero_Curry_basis_functions(T = T1, p = p1, t = t1, normalize = normalize)
                D2 = non_zero_Curry_basis_functions(T = T2, p = p2, t = t2, normalize = normalize)
                D3 = non_zero_Curry_basis_functions(T = T3, p = p3, t = t3, normalize = normalize)
                
                B1 = csr_matrix(B1)
                B2 = csr_matrix(B2)
                B3 = csr_matrix(B3)
                D1 = csr_matrix(D1)
                D2 = csr_matrix(D2)
                D3 = csr_matrix(D3)
            
                vol1 += sp_kron(sp_kron(B1,D2),D3)*f(t1, t2, t3)[0]*w1[s1]*w2[s2]*w3[s3]
                vol2 += sp_kron(sp_kron(D1,B2),D3)*f(t1, t2, t3)[1]*w1[s1]*w2[s2]*w3[s3]
                vol3 += sp_kron(sp_kron(D1,D2),B3)*f(t1, t2, t3)[2]*w1[s1]*w2[s2]*w3[s3]
           
    vol1 = 0.125*h1*h2*h3*vol1 
    vol2 = 0.125*h1*h2*h3*vol2 
    vol3 = 0.125*h1*h2*h3*vol3 
    
    vol1 = vol1.toarray()
    vol2 = vol2.toarray()
    vol3 = vol3.toarray()
    
    vol1 = vol1.reshape((p1+1, p2, p3))
    vol2 = vol2.reshape((p1, p2+1, p3))
    vol3 = vol3.reshape((p1, p2, p3+1))
    
    return [vol1, vol2, vol3]


# ========================================================================
def div_elementary_vectors(N, p, f, normalize=True,  separated_variables = False):
    
    (N1, N2, N3) = N
    
    global elementary_vector_function
    
    def elementary_vector_function(e1, e2, e3) : 
        return div_rhs_elementary_vector(e1, e2, e3, N, p, f, 
                                          normalize           = normalize, 
                                          separated_variables = separated_variables)  
    
    
    
    process_pool = Pool(3)
    data = [(e1, e2, e3) for e1 in range(N1) for e2 in range(N2) for e3 in range(N3)]
    elementary_vectors = process_pool.starmap(elementary_vector_function , data)
    
    return elementary_vectors
    
    
# ========================================================================
def div_rhs_vector(N, p, f, normalize=True, separated_variables = True):
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0], p1)
    T2           = knot_vector(N[1], p2)
    T3           = knot_vector(N[2], p3)
    
    if separated_variables == True:
        print('TODO A SIMPLE IMPLEMENTATION IN THE CASE WHERE A FUNCTION WITH SEPARATED VARIABLES IS GIVEN')
    
    else:
        (N1, N2, N3)                   = N
        (m1, m2, m3)                   = (len(T1), len(T2), len(T3))
        (n1, n2, n3)                   = (m1-p1-1, m2-p2-1, m3-p3-1)
        
        rhs1                           = np.zeros((n1, n2-1, n3-1), dtype = float)
        rhs2                           = np.zeros((n1-1, n2, n3-1), dtype = float)
        rhs3                           = np.zeros((n1-1, n2-1, n3), dtype = float)
        
        elementary_vectors             = div_elementary_vectors(N, p, f, 
                                                                 normalize           = normalize,  
                                                                 separated_variables = separated_variables)
        
        
        for e1 in range(N1):
            i1_l = location_matrix(T=T1, p=p1)[e1,:][0]
            i1_r = location_matrix(T=T1, p=p1)[e1,:][-1]+1
            j1_l = location_matrix_2(T=T1, p=p1)[e1,:][0]
            j1_r = location_matrix_2(T=T1, p=p1)[e1,:][-1]+1
            k1_l = location_matrix_2(T=T1, p=p1)[e1,:][0]
            k1_r = location_matrix_2(T=T1, p=p1)[e1,:][-1]+1
            
            for e2 in range(N2):
                
                i2_l = location_matrix_2(T=T2, p=p2)[e2,:][0]
                i2_r = location_matrix_2(T=T2, p=p2)[e2,:][-1]+1
                j2_l = location_matrix(T=T2, p=p2)[e2,:][0]
                j2_r = location_matrix(T=T2, p=p2)[e2,:][-1]+1
                k2_l = location_matrix_2(T=T2, p=p2)[e2,:][0]
                k2_r = location_matrix_2(T=T2, p=p2)[e2,:][-1]+1
                
                for e3 in range(N3):
                    i3_l = location_matrix_2(T=T3, p=p3)[e3,:][0]
                    i3_r = location_matrix_2(T=T3, p=p3)[e3,:][-1]+1
                    j3_l = location_matrix_2(T=T3, p=p3)[e3,:][0]
                    j3_r = location_matrix_2(T=T3, p=p3)[e3,:][-1]+1
                    k3_l = location_matrix(T=T3, p=p3)[e3,:][0]
                    k3_r = location_matrix(T=T3, p=p3)[e3,:][-1]+1
                    
                    [vol1, vol2, vol3] = elementary_vectors[e3+e2*N3+e1*N2*N3]
                    
                    rhs1[i1_l:i1_r, i2_l:i2_r, i3_l:i3_r] += vol1
                    rhs2[j1_l:j1_r, j2_l:j2_r, j3_l:j3_r] += vol2
                    rhs3[k1_l:k1_r, k2_l:k2_r, k3_l:k3_r] += vol3
        
        rhs1 = rhs1[1:-1,:, :] 
        rhs2 = rhs2[:, 1:-1, :]
        rhs3 = rhs3[:, :, 1:-1]            
                    
        rhs1 = rhs1.flatten()
        rhs2 = rhs2.flatten()
        rhs3 = rhs3.flatten()
    
        rhs = np.hstack((np.hstack((rhs1, rhs2)), rhs3))
        
    return rhs                     
    
# ========================================================================
def div_appr_sol_function(N, p, u, x, normalize=True):
    
    (p1, p2, p3) = (p[0], p[1], p[2])
    T1           = knot_vector(N[0],p1)
    T2           = knot_vector(N[1],p2)
    T3           = knot_vector(N[2],p3)
    
    if len(u)%3 == 0:
        u1 = u[:len(u)//3]
        u2 = u[len(u)//3:2*(len(u)//3)]
        u3 = u[2*(len(u)//3):]
    else:
        raise NotImplementedError('')
        
    (x1, x2, x3) = x
    
    b1 = basis_functions(T1, p1, x1)[1:-1]
    b2 = basis_functions(T2, p2, x2)[1:-1]
    b3 = basis_functions(T3, p3, x3)[1:-1]
    d1 = Curry_basis_functions(T1,p1,x1, normalize=normalize)
    d2 = Curry_basis_functions(T2,p2,x2, normalize=normalize)
    d3 = Curry_basis_functions(T3,p3,x3, normalize=normalize)
    
    c1 = sp_kron(sp_kron(b1,d2),d3)
    c2 = sp_kron(sp_kron(d1,b2),d3) 
    c3 = sp_kron(sp_kron(d1,d2),b3)
    
    c1 = np.squeeze(c1.toarray())
    c2 = np.squeeze(c2.toarray())
    c3 = np.squeeze(c3.toarray())

    uh1 = np.dot(u1, c1)
    uh2 = np.dot(u2, c2)
    uh3 = np.dot(u3, c3)
    
    uh = (uh1, uh2, uh3)
    
    return uh       
    
def div_DeRham_projection_coefficients_vector(N, p, u, normalize=True, separated_variables = False):
    
    (p1,p2,p3) = (p[0],p[1],p[2])
    T1         = knot_vector(N[0],p1)
    T2         = knot_vector(N[1],p2)
    T3         = knot_vector(N[2],p3)
    
    if separated_variables == True:
        print('TODO A SIMPLE IMPLEMENTATION IN THE CASE WHERE A FUNCTION WITH SEPARATED VARIABLES IS GIVEN')
    
    else:
        (N1, N2, N3)             = N
        (m1, m2, m3)             = (len(T1), len(T2), len(T3))
        (n1, n2, n3)             = (m1-p1-1, m2-p2-1, m3-p3-1)
        (ordergl1, ordergl2, ordergl3) = (p1+1, p2+1, p3+1)
        (r1, w1)             = Gauss_Legendre(ordergl = ordergl1,
                                              tol     = 10e-14) 
        (r2, w2)             = Gauss_Legendre(ordergl = ordergl2,
                                              tol     = 10e-14) 
        (r3, w3)             = Gauss_Legendre(ordergl = ordergl3,
                                              tol     = 10e-14) 
        (k1, k2, k3)             = (len(r1), len(r2), len(r3))
        
        g1                   = Greville_points(T=T1, p=p1)
        g2                   = Greville_points(T=T2, p=p2)
        g3                   = Greville_points(T=T3, p=p3)
    
        y1                   = np.zeros((n1, n2-1, n3-1), dtype = float)
        y2                   = np.zeros((n1-1, n2, n3-1), dtype = float)
        y3                   = np.zeros((n1-1, n2-1, n3), dtype = float)
        
        for i1 in range(n1):
            for i2 in range(n2-1):
                a2 = g2[i2]
                b2 = g2[i2+1]
                h2 = b2-a2
                for i3 in range(n3-1): 
                    a3 = g3[i3]
                    b3 = g3[i3+1]
                    h3 = b3-a3
                    vol = 0.0
                    for s2 in range(k2):
                        t2 = 0.5*(h2*r2[s2]+a2+b2)
                        for s3 in range(k3):
                            t3 = 0.5*(h3*r3[s3]+a3+b3)
                            vol += u(g1[i1], t2, t3)[0]*w2[s2]*w3[s3]
                    y1[i1][i2][i3] = 0.25*h2*h3*vol
        
        for i1 in range(n1-1):
            a1 = g1[i1]
            b1 = g1[i1+1]
            h1 = b1-a1
            for i2 in range(n2):
                for i3 in range(n3-1): 
                    a3 = g3[i3]
                    b3 = g3[i3+1]
                    h3 = b3-a3
                    vol = 0.0
                    for s1 in range(k1):
                        t1 = 0.5*(h1*r1[s1]+a1+b1)
                        for s3 in range(k3):
                            t3 = 0.5*(h3*r3[s3]+a3+b3)
                            vol += u(t1, g2[i2], t3)[1]*w1[s1]*w3[s3]
                    y2[i1][i2][i3] = 0.25*h1*h3*vol
        
        for i1 in range(n1-1):
            a1 = g1[i1]
            b1 = g1[i1+1]
            h1 = b1-a1
            for i2 in range(n2-1):
                a2 = g2[i2]
                b2 = g2[i2+1]
                h2 = b2-a2
                for i3 in range(n3): 
                    vol = 0.0
                    for s1 in range(k1):
                        t1 = 0.5*(h1*r1[s1]+a1+b1)
                        for s2 in range(k2):
                            t2 = 0.5*(h2*r2[s2]+a2+b2)
                            vol += u(t1, t2, g3[i3])[2]*w1[s1]*w2[s2]
                    y3[i1][i2][i3] = 0.25*h1*h2*vol
                    
        y1 = y1[1:-1, :, :].flatten()
        y2 = y2[:, 1:-1, :].flatten()
        y3 = y3[:, :, 1:-1].flatten()
        
        M1 = collocation_matrix(T1, p1)[1:-1, 1:-1]
        M2 = collocation_matrix(T2, p2)[1:-1, 1:-1]
        M3 = collocation_matrix(T3, p3)[1:-1, 1:-1]
        
        H1 = histopolation_matrix(T1, p1, normalize=normalize)
        H2 = histopolation_matrix(T2, p2, normalize=normalize)
        H3 = histopolation_matrix(T3, p3, normalize=normalize)
        
        P1 = sp_kron(sp_kron(M1, H2), H3)
        P2 = sp_kron(sp_kron(H1, M2), H3)
        P3 = sp_kron(sp_kron(H1, H2), M3)
        
        P1 = P1.toarray()
        P2 = P2.toarray()
        P3 = P3.toarray()
    
        c1 = la.solve(P1, y1)
        c2 = la.solve(P2, y2)
        c3 = la.solve(P3, y3)
        
        c  = np.hstack((c1, c2, c3))
        return c
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

