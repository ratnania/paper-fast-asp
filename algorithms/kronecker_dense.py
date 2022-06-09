"""
implements kron product in the case of CSR matrices
"""

from pyccel.stdlib.internal.blas import dgemm


def blas_dense_product(A:'float[:,:]', x:'float[:,:]', y:'float[:,:]'):
    
    import numpy as np
    
    row_A = np.int32(A.shape[0])
    col_A = np.int32(A.shape[1])
    
    row_x = np.int32(x.shape[0])
    col_x = np.int32(x.shape[1])
    
    xx = np.zeros((x.shape[1], x.shape[0]), order="F")
    c = np.zeros((A.shape[0], xx.shape[1]), order="F")
    
    col_xx = np.int32(xx.shape[1])
    
    alpha = 1.0
    beta  = 0.0
    
    #$ omp for schedule(runtime)
    for j in range(col_x):
          for i in range(row_x):
                xx[j,i] = x[i,j]
                
    a = np.array(A, order='F')
    dgemm('N', 'N', row_A, col_xx, col_A, alpha, a, row_A, xx, col_A, beta, c, row_A)
    
    y[:,:] = c[:,:]
    
# =========================================================================
def vec_2d_omp(x_mat: 'float[:,:]', x: 'float[:]'):
    """Convert a matrix to a vector form."""

    n1, n2 = x_mat.shape

    #$ omp for schedule(runtime) collapse(2)
    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x[i] = x_mat[i1, i2]

# =========================================================================
def unvec_2d_omp(x: 'float[:]', n1: int, n2: int, x_mat: 'float[:,:]'):
    """Convert a vector to a matrix form."""

    #$ omp for schedule(runtime) collapse(2)
    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x_mat[i1, i2] = x[i]

# =========================================================================
def kron_2d(A1: 'float[:,:]', A2: 'float[:,:]', 
            n_rows_1: int, n_cols_1: int,
            n_rows_2: int, n_cols_2: int,
            x: 'float[:]', 
            W1: 'float[:,:]', W2: 'float[:,:]', 
            y: 'float[:]'):
    # ...
    unvec_2d_omp(x, n_cols_1, n_cols_2, W1[0:n_cols_1, 0:n_cols_2])
    
    blas_dense_product(A2, W1[:n_cols_1, 0:n_cols_2], W2[:n_rows_2,:n_cols_1])
    
    # ...
    blas_dense_product(A1, W2[:n_rows_2,:n_cols_1], W1[:n_rows_1,:n_rows_2])
    # ...

    # ...
    vec_2d_omp(W1[0:n_rows_1, 0:n_rows_2], y)
    # ...

# =========================================================================
def kron_3d(A1: 'float[:,:]', A2: 'float[:,:]', A3: 'float[:,:]',
            n_rows_1: int, n_cols_1: int,
            n_rows_2: int, n_cols_2: int,
            n_rows_3: int, n_cols_3: int,
            x: 'float[:]',
            Z1: 'float[:,:]', Z2: 'float[:,:]',
            Z3: 'float[:,:]', Z4: 'float[:,:]',
            y: 'float[:]'):
    
    #$ omp parallel

    n_rows_12 = n_rows_1 * n_rows_2
    n_cols_12 = n_cols_1 * n_cols_2
    # ...

    unvec_2d_omp(x, n_cols_12, n_cols_3, Z1[0:n_cols_12, 0:n_cols_3])
   
    blas_dense_product(A3, Z1[:n_cols_12,:n_cols_3], Z2[:n_rows_3,:n_cols_12])

    #$ omp for schedule(runtime) private(k, Z3, Z4)
    for k in range(n_rows_3):
          kron_2d(A1, A2,
                  n_rows_1, n_cols_1,
                  n_rows_2, n_cols_2,
                  Z2[k,0:n_cols_12],
                  Z3, Z4,
                  Z1[0:n_rows_12,k])


    vec_2d_omp(Z1[0:n_rows_12,0:n_rows_3], y)

    #$ omp end parallel