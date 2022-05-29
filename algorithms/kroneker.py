  
# def multi_index():
#     pass
import numpy as np
from scipy.sparse import kron as sp_kron

def kroneker_lower(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
                   A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
                   A3_data: 'float[:]', A3_ind: 'int32[:]', A3_ptr: 'int32[:]',
                   B1_data: 'float[:]', B1_ind: 'int32[:]', B1_ptr: 'int32[:]',
                   B2_data: 'float[:]', B2_ind: 'int32[:]', B2_ptr: 'int32[:]',
                   B3_data: 'float[:]', B3_ind: 'int32[:]', B3_ptr: 'int32[:]',
                   C1_data: 'float[:]', C1_ind: 'int32[:]', C1_ptr: 'int32[:]',
                   C2_data: 'float[:]', C2_ind: 'int32[:]', C2_ptr: 'int32[:]',
                   C3_data: 'float[:]', C3_ind: 'int32[:]', C3_ptr: 'int32[:]',
                   alpha: float, beta: float, gamma: float,
                   b: 'float[:]', x: 'float[:]'):
    
    n1 = len(A1_ptr) - 1 # -1 because A1_ptr[i1+1] access to index i1+1
    n2 = len(A2_ptr) - 1 # -1 because A2_ptr[i2+1] access to index i2+1
    n3 = len(A3_ptr) - 1 # -1 because A3_ptr[i3+1] access to index i3+1
    
    for i1 in range(1, n1):
        for i2 in range(1, n2):
            for i3 in range(1, n3):
                i = i3 + (i2 + i1 * n2) * n3#multi_index(i1,i2,i3)
                yi = 0.
                ad = 1.
                for k1 in range(A1_ptr[i1], A1_ptr[i1+1]):
                    j1 = A1_ind[k1]
                    a1 = A1_data[k1]
                    for k2 in range(A2_ptr[i2], A2_ptr[i2+1]):
                        j2 = A2_ind[k2]
                        a2 = A2_data[k2]
                        for k3 in range(A3_ptr[i3], A3_ptr[i3+1]):
                            j3 = A3_ind[k3]
                            a3 = A3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i < j:
                                yi = yi + a1*a2*a3*x[j]
                            else:
                                ad = a1*a2*a3
                zi = 0.
                bd = 1.
                for k1 in range(B1_ptr[i1], B1_ptr[i1+1]):
                    j1 = B1_ind[k1]
                    a1 = B1_data[k1]
                    for k2 in range(B2_ptr[i2], B2_ptr[i2+1]):
                        j2 = B2_ind[k2]
                        a2 = B2_data[k2]
                        for k3 in range(B3_ptr[i3], B3_ptr[i3+1]):
                            j3 = B3_ind[k3]
                            a3 = B3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i < j:
                                zi = zi + a1*a2*a3*x[j]
                            else:
                                bd = a1*a2*a3
                wi = 0.
                cd = 1.
                for k1 in range(C1_ptr[i1], C1_ptr[i1+1]):
                    j1 = C1_ind[k1]
                    a1 = C1_data[k1]
                    for k2 in range(C2_ptr[i2], C2_ptr[i2+1]):
                        j2 = C2_ind[k2]
                        a2 = C2_data[k2]
                        for k3 in range(C3_ptr[i3], C3_ptr[i3+1]):
                            j3 = C3_ind[k3]
                            a3 = C3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i < j:
                                wi = wi + a1*a2*a3*x[j]
                            else:
                                cd = a1*a2*a3
                                
                                j = j3 + (j2 + j1 * n2) * n3
                           
                x[i] = 1/(alpha*ad + beta*bd*gamma*cd) * (b[i] - alpha*yi - beta*zi - gamma*wi)
    # return x


def kroneker_upper(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
                   A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
                   A3_data: 'float[:]', A3_ind: 'int32[:]', A3_ptr: 'int32[:]',
                   B1_data: 'float[:]', B1_ind: 'int32[:]', B1_ptr: 'int32[:]',
                   B2_data: 'float[:]', B2_ind: 'int32[:]', B2_ptr: 'int32[:]',
                   B3_data: 'float[:]', B3_ind: 'int32[:]', B3_ptr: 'int32[:]',
                   C1_data: 'float[:]', C1_ind: 'int32[:]', C1_ptr: 'int32[:]',
                   C2_data: 'float[:]', C2_ind: 'int32[:]', C2_ptr: 'int32[:]',
                   C3_data: 'float[:]', C3_ind: 'int32[:]', C3_ptr: 'int32[:]',
                   alpha: float, beta: float, gamma: float,
                   b: 'float[:]', x: 'float[:]'):
    
    n1 = len(A1_ptr) - 1 # -1 because A1_ptr[i1+1] access to index i1+1
    n2 = len(A2_ptr) - 1 # -1 because A2_ptr[i2+1] access to index i2+1
    n3 = len(A3_ptr) - 1 # -1 because A3_ptr[i3+1] access to index i3+1
    
    for i1 in range(1, n1):
        for i2 in range(1, n2):
            for i3 in range(1, n3):
                i = i3 + (i2 + i1 * n2) * n3#multi_index(i1,i2,i3)
                
                yi = 0.
                ad = 1.
                for k1 in range(A1_ptr[i1], A1_ptr[i1+1]):
                    j1 = A1_ind[k1]
                    a1 = A1_data[k1]
                    for k2 in range(A2_ptr[i2], A2_ptr[i2+1]):
                        j2 = A2_ind[k2]
                        a2 = A2_data[k2]
                        for k3 in range(A3_ptr[i3], A3_ptr[i3+1]):
                            j3 = A3_ind[k3]
                            a3 = A3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i >= j:
                                yi = yi + a1*a2*a3*x[j]
                            if i == j:
                                ad = a1*a2*a3
                zi = 0.
                bd = 1.
                for k1 in range(B1_ptr[i1], B1_ptr[i1+1]):
                    j1 = B1_ind[k1]
                    a1 = B1_data[k1]
                    for k2 in range(B2_ptr[i2], B2_ptr[i2+1]):
                        j2 = B2_ind[k2]
                        a2 = B2_data[k2]
                        for k3 in range(B3_ptr[i3], B3_ptr[i3+1]):
                            j3 = B3_ind[k3]
                            a3 = B3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i >= j:
                                zi = zi + a1*a2*a3*x[j]
                            if i == j:
                                bd = a1*a2*a3
                wi = 0.
                cd = 1.
                for k1 in range(C1_ptr[i1], C1_ptr[i1+1]):
                    j1 = C1_ind[k1]
                    a1 = C1_data[k1]
                    for k2 in range(C2_ptr[i2], C2_ptr[i2+1]):
                        j2 = C2_ind[k2]
                        a2 = C2_data[k2]
                        for k3 in range(C3_ptr[i3], C3_ptr[i3+1]):
                            j3 = C3_ind[k3]
                            a3 = C3_data[k3]
                            j =  j3 + (j2 + j1 * n2) * n3#multi_index(j1,j2,j3)
                            if i >= j:
                                wi = wi + a1*a2*a3*x[j]
                            if i == j:
                                cd = a1*a2*a3
                                
                                j = j3 + (j2 + j1 * n2) * n3
                            
                            
                x[i] = 1/(alpha*ad + beta*bd*gamma*cd) * (b[i] - alpha*yi - beta*zi - gamma*wi)
                
    # return x
# =========================================================================



# =========================================================================
def spsolve_kron_csr_3_sum_lower(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
                                 A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
                                 A3_data: 'float[:]', A3_ind: 'int32[:]', A3_ptr: 'int32[:]',
                                 B1_data: 'float[:]', B1_ind: 'int32[:]', B1_ptr: 'int32[:]',
                                 B2_data: 'float[:]', B2_ind: 'int32[:]', B2_ptr: 'int32[:]',
                                 B3_data: 'float[:]', B3_ind: 'int32[:]', B3_ptr: 'int32[:]',
                                 C1_data: 'float[:]', C1_ind: 'int32[:]', C1_ptr: 'int32[:]',
                                 C2_data: 'float[:]', C2_ind: 'int32[:]', C2_ptr: 'int32[:]',
                                 C3_data: 'float[:]', C3_ind: 'int32[:]', C3_ptr: 'int32[:]',
                                 alpha: float, beta: float, gamma: float,
                                 b: 'float[:]', y: 'float[:]'):

    n1 = len(A1_ptr) - 1
    n2 = len(A2_ptr) - 1
    n3 = len(A3_ptr) - 1
    n = n1 * n2 * n3

    for i in range(n):

        i1 = i // ( n2 * n3 )
        r  = i - i1 * n2 * n3
        i2 = r // n3
        i3 = r - i2 * n3

        # ...
        k1_b = A1_ptr[i1]
        k1_e = A1_ptr[i1+1]
        k2_b = A2_ptr[i2]
        k2_e = A2_ptr[i2+1]
        k3_b = A3_ptr[i3]
        k3_e = A3_ptr[i3+1]

        yi = 0.
        a_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = A1_ind[k1]
            a1 = A1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = A2_ind[k2]
                a2 = A2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = A3_ind[k3]
                    a3 = A3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j < i:
                        yi += a1 * a2 * a3 * y[j]

                    elif i == j:
                        a_d = a1 * a2 * a3
        # ...

        # ...
        k1_b = B1_ptr[i1]
        k1_e = B1_ptr[i1+1]
        k2_b = B2_ptr[i2]
        k2_e = B2_ptr[i2+1]
        k3_b = B3_ptr[i3]
        k3_e = B3_ptr[i3+1]

        zi = 0.
        b_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = B1_ind[k1]
            a1 = B1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = B2_ind[k2]
                a2 = B2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = B3_ind[k3]
                    a3 = B3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j < i:
                        zi += a1 * a2 * a3 * y[j]

                    elif i == j:
                        b_d = a1 * a2 * a3
        # ...

        # ...
        k1_b = C1_ptr[i1]
        k1_e = C1_ptr[i1+1]
        k2_b = C2_ptr[i2]
        k2_e = C2_ptr[i2+1]
        k3_b = C3_ptr[i3]
        k3_e = C3_ptr[i3+1]

        wi = 0.
        c_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = C1_ind[k1]
            a1 = C1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = C2_ind[k2]
                a2 = C2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = C3_ind[k3]
                    a3 = C3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j < i:
                        wi += a1 * a2 * a3 * y[j]

                    elif i == j:
                        c_d = a1 * a2 * a3
                        
                    
        # ...

        y[i] = ( b[i] - alpha * yi - beta * zi - gamma * wi ) / ( alpha * a_d + beta * b_d + gamma * c_d )

# =========================================================================
def spsolve_kron_csr_3_sum_upper(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
                                 A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
                                 A3_data: 'float[:]', A3_ind: 'int32[:]', A3_ptr: 'int32[:]',
                                 B1_data: 'float[:]', B1_ind: 'int32[:]', B1_ptr: 'int32[:]',
                                 B2_data: 'float[:]', B2_ind: 'int32[:]', B2_ptr: 'int32[:]',
                                 B3_data: 'float[:]', B3_ind: 'int32[:]', B3_ptr: 'int32[:]',
                                 C1_data: 'float[:]', C1_ind: 'int32[:]', C1_ptr: 'int32[:]',
                                 C2_data: 'float[:]', C2_ind: 'int32[:]', C2_ptr: 'int32[:]',
                                 C3_data: 'float[:]', C3_ind: 'int32[:]', C3_ptr: 'int32[:]',
                                 alpha: float, beta: float, gamma: float,
                                 b: 'float[:]', y: 'float[:]'):

    n1 = len(A1_ptr) - 1
    n2 = len(A2_ptr) - 1
    n3 = len(A3_ptr) - 1
    n = n1 * n2 * n3

    for i in range(n-1,-1,-1):

        i1 = i // ( n2 * n3 )
        r  = i - i1 * n2 * n3
        i2 = r // n3
        i3 = r - i2 * n3

        # ...
        k1_b = A1_ptr[i1]
        k1_e = A1_ptr[i1+1]
        k2_b = A2_ptr[i2]
        k2_e = A2_ptr[i2+1]
        k3_b = A3_ptr[i3]
        k3_e = A3_ptr[i3+1]

        yi = 0.
        a_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = A1_ind[k1]
            a1 = A1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = A2_ind[k2]
                a2 = A2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = A3_ind[k3]
                    a3 = A3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j >= i:
                        yi += a1 * a2 * a3 * y[j]

                    if i == j:
                        a_d = a1 * a2 * a3
        # ...

        # ...
        k1_b = B1_ptr[i1]
        k1_e = B1_ptr[i1+1]
        k2_b = B2_ptr[i2]
        k2_e = B2_ptr[i2+1]
        k3_b = B3_ptr[i3]
        k3_e = B3_ptr[i3+1]

        zi = 0.
        b_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = B1_ind[k1]
            a1 = B1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = B2_ind[k2]
                a2 = B2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = B3_ind[k3]
                    a3 = B3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j >= i:
                        zi += a1 * a2 * a3 * y[j]

                    if i == j:
                        b_d = a1 * a2 * a3
        # ...

        # ...
        k1_b = C1_ptr[i1]
        k1_e = C1_ptr[i1+1]
        k2_b = C2_ptr[i2]
        k2_e = C2_ptr[i2+1]
        k3_b = C3_ptr[i3]
        k3_e = C3_ptr[i3+1]

        wi = 0.
        c_d = 1.
        for k1 in range(k1_b, k1_e):
            j1 = C1_ind[k1]
            a1 = C1_data[k1]

            for k2 in range(k2_b, k2_e):
                j2 = C2_ind[k2]
                a2 = C2_data[k2]

                for k3 in range(k3_b, k3_e):
                    j3 = C3_ind[k3]
                    a3 = C3_data[k3]

                    j = j3 + (j2 + j1 * n2) * n3
                    if j >= i:
                        wi += a1 * a2 * a3 * y[j]

                    if i == j:
                        c_d = a1 * a2 * a3
        # ...

        y[i] = ( b[i] - alpha * yi - beta * zi - gamma * wi ) / ( alpha * a_d + beta * b_d + gamma * c_d )
        
        
        
# =========================================================================
def spsolve_kron_sum(matrices, confficients, b, lower):
    
    A1, A2, A3, B1, B2, B3, C1, C2, C3 = matrices
    alpha, beta, gamma                 = confficients
    
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
    
    N = np.shape(A1)[0]*np.shape(A2)[0]*np.shape(A3)[0]
    out = np.zeros(N)
    
    
    if lower == True:
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
                                     b, out)
    elif lower == False:
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
                                     b, out)
    else:
        raise NotImplementedError('lower =True or False!!!!')
    
    return out
    
# =========================================================================
def spsolve_kron_block(matrices, confficients, b):
    """
    Solves a triangular system of the form

    | A11   0   0 | [x1] = [y1]
    | A21 A22   0 | [x2] = [y2]
    | A31 A32 A33 | [x3] = [y3]

    if A11 and A22 and A33 are lower triangular matrices.

    or

    | A11 A12 A13 | [x1] = [y1]
    |   0 A22 A23 | [x2] = [y2]
    |   0   0 A33 | [x3] = [y3]

    if A11 and A22 and A33 are upper triangular matrices.
    0 matrices must be specified as None
    """
    # ...
    
    A11 = matrices[0] 
    A12 = matrices[1] 
    A13 = matrices[2] 
    A21 = matrices[3] 
    A22 = matrices[4] 
    A23 = matrices[5] 
    A31 = matrices[6] 
    A32 = matrices[7] 
    A33 = matrices[8] 
    
    c1 = confficients[0] 
    c2 = confficients[1]
    c3 = confficients[2] 
    
    
    A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
    A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
    A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33
    

    n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
    n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
    n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0]
    # ...
    
    # ...
    b1 = b[:n1]
    b2 = b[n1:n1+n2]
    b3 = b[n1+n2:]
    # ...

    # lower triangular matrix case
    if (A12 is None) and (A13 is None) and (A23 is None):
        x1 = spsolve_kron_sum(A11, c1, b1, lower=True)
        x2 = spsolve_kron_sum(A22, c2, b2 - A21.dot(x1), lower=True)
        x3 = spsolve_kron_sum(A33, c3, b3 - A31.dot(x1) - A32.dot(x2), lower=True)

    # upper triangular matrix case
    elif (A21 is None) and (A31 is None) and (A32 is None):
        x3 = spsolve_kron_sum(A33, c3, b3, lower=False)
        x2 = spsolve_kron_sum(A22, c2, b2 - A23.dot(x3), lower=False)
        x1 = spsolve_kron_sum(A11, c1, b1 - A12.dot(x2) - A13.dot(x3), lower=False)

    else:
        raise ValueError('Wrong entries')

    return np.concatenate([x1,x2,x3])

def Gauss_Seidel(matrices,  confficients, b, x, iterations_number):
    
    A11 = matrices[0] 
    A12 = matrices[1] 
    A13 = matrices[2] 
    A21 = matrices[3] 
    A22 = matrices[4] 
    A23 = matrices[5] 
    A31 = matrices[6] 
    A32 = matrices[7] 
    A33 = matrices[8] 
    
    c1 = confficients[0] 
    c2 = confficients[1]
    c3 = confficients[2] 
    
    
    A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
    A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
    A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33
    

    n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
    n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
    n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0]
    # ...
    
    # ...
    b1 = b[:n1]
    b2 = b[n1:n1+n2]
    b3 = b[n1+n2:]
    # ...
    
    (D1, M2, K3, D1, K2, M3, D1, M2, M3) = A11
    (K1, D2, M3, M1, D2, K3, M1, D2, M3) = A22
    (M1, K2, D3, K1, M2, D3, M1, M2, D3) = A33

    M11 = c1[0]*sp_kron(sp_kron(D1,M2),K3) + c1[1]*sp_kron(sp_kron(D1,K2),M3) + c1[2]*sp_kron(sp_kron(D1,M2),M3)
    M22 = c2[0]*sp_kron(sp_kron(K1,D2),M3) + c2[1]*sp_kron(sp_kron(M1,D2),K3) + c2[2]*sp_kron(sp_kron(M1,D2),M3) 
    M33 = c3[0]*sp_kron(sp_kron(M1,K2),D3) + c3[1]*sp_kron(sp_kron(K1,M2),D3) + c3[2]*sp_kron(sp_kron(M1,M2),D3) 
    

    
    
    if x == None:
        x = np.zeros(n1+n2+n3)
    #A = csr_matrix(A)
    m = iterations_number
    #x = x0
    for i in range(m):
        x1 = x[:n1]
        x2 = x[n1:n1+n2]
        x3 = x[n1+n2:]
        
        B1 = M11.dot(x1)+A12.dot(x2)+A13.dot(x3) 
        B2 = A21.dot(x1)+M22.dot(x2)+A23.dot(x3) 
        B3 = A31.dot(x1)+A32.dot(x2)+M33.dot(x3)
        
        B = np.concatenate([B1,B2,B3])
        
        A = [A11, None, None,
             A21, A22, None,
             A31, A32, A33]
        x += spsolve_kron_block(A, confficients, b-B)
        
    for i in range(m):
        x1 = x[:n1]
        x2 = x[n1:n1+n2]
        x3 = x[n1+n2:]
        
        B1 = M11.dot(x1)+A12.dot(x2)+A13.dot(x3) 
        B2 = A21.dot(x1)+M22.dot(x2)+A23.dot(x3) 
        B3 = A31.dot(x1)+A32.dot(x2)+M33.dot(x3)
        
        B = np.concatenate([B1,B2,B3])
        
        A = [A11, A12, A13,
             None, A22, A23,
             None, None, A33]
        x += spsolve_kron_block(A, confficients, b-B)
    return x


