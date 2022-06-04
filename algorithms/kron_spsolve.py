
#from scipy.sparse import kron as sp_kron
import numpy as np

from kroneker import spsolve_kron_csr_3_sum_lower, spsolve_kron_csr_3_sum_upper
from kron_product import kron_product_csr

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
    
    n1 = A11[0].shape[0]*A11[1].shape[0]*A11[2].shape[0]
    n2 = A22[0].shape[0]*A22[1].shape[0]*A22[2].shape[0]
    #n3 = A33[0].shape[0]*A33[1].shape[0]*A33[2].shape[0]
    # ...
    
    # ...
    b1 = b[:n1]
    b2 = b[n1:n1+n2]
    b3 = b[n1+n2:]
    # ...
    
    # lower triangular matrix case
    if (A12 is None) and (A13 is None) and (A23 is None):
        x1 = spsolve_kron_sum(A11, c1, b1, lower=True)
        
        w1 = kron_product_csr(A21, x1)
        x2 = spsolve_kron_sum(A22, c2, b2 - w1, lower=True)
        
        w2 = kron_product_csr(A31, x1)+kron_product_csr(A32, x2)
        x3 = spsolve_kron_sum(A33, c3, b3 - w2, lower=True)

    # upper triangular matrix case
    elif (A21 is None) and (A31 is None) and (A32 is None):
        x3 = spsolve_kron_sum(A33, c3, b3, lower=False)
        
        w2 = kron_product_csr(A23, x3)
        x2 = spsolve_kron_sum(A22, c2, b2 - w2, lower=False)
        
        w1 = kron_product_csr(A12, x2)+kron_product_csr(A13, x3)
        x1 = spsolve_kron_sum(A11, c1, b1 - w1, lower=False)

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
    
    
    n1 = A11[0].shape[0]*A11[1].shape[0]*A11[2].shape[0]
    n2 = A22[0].shape[0]*A22[1].shape[0]*A22[2].shape[0]
    n3 = A33[0].shape[0]*A33[1].shape[0]*A33[2].shape[0]
    # ...
     
    if x == None:
        x = np.zeros(n1+n2+n3)
    #A = csr_matrix(A)
    m = iterations_number
    #x = x0
    for i in range(m):
        x1 = x[:n1]
        x2 = x[n1:n1+n2]
        x3 = x[n1+n2:]
        
        B1 = c1[0]*kron_product_csr(A11[:3], x1) + c1[1]*kron_product_csr(A11[3:6], x1)\
            + c1[2]*kron_product_csr(A11[6:], x1) + kron_product_csr(A12, x2)\
                 + kron_product_csr(A13, x3)
        
        B2 = kron_product_csr(A21, x1) + c2[0]*kron_product_csr(A22[:3], x2)\
            + c2[1]*kron_product_csr(A22[3:6], x2) + c2[2]*kron_product_csr(A22[6:], x2)\
                + kron_product_csr(A23, x3)
        
        B3 = kron_product_csr(A31, x1) + kron_product_csr(A32, x2)\
            + c3[0]*kron_product_csr(A33[:3], x3) + c3[1]*kron_product_csr(A33[3:6], x3)\
                + c3[2]*kron_product_csr(A33[6:], x3)\
            
        B = np.concatenate([B1,B2,B3])
        
        A = [A11, None, None,
             A21, A22, None,
             A31, A32, A33]
        x += spsolve_kron_block(A, confficients, b-B)
        
    for i in range(m):
        x1 = x[:n1]
        x2 = x[n1:n1+n2]
        x3 = x[n1+n2:]
        
        B1 = c1[0]*kron_product_csr(A11[:3], x1) + c1[1]*kron_product_csr(A11[3:6], x1)\
            + c1[2]*kron_product_csr(A11[6:], x1) + kron_product_csr(A12, x2)\
                 + kron_product_csr(A13, x3)
        
        B2 = kron_product_csr(A21, x1) + c2[0]*kron_product_csr(A22[:3], x2)\
            + c2[1]*kron_product_csr(A22[3:6], x2) + c2[2]*kron_product_csr(A22[6:], x2)\
                + kron_product_csr(A23, x3)
        
        B3 = kron_product_csr(A31, x1) + kron_product_csr(A32, x2)\
            + c3[0]*kron_product_csr(A33[:3], x3) + c3[1]*kron_product_csr(A33[3:6], x3)\
                + c3[2]*kron_product_csr(A33[6:], x3)\
            
        B = np.concatenate([B1,B2,B3])
        
        A = [A11, A12, A13,
             None, A22, A23,
             None, None, A33]
        x += spsolve_kron_block(A, confficients, b-B)
    return x


