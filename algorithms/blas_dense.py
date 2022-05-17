from pyccel.stdlib.internal.blas import dgemv


def blas_dense_product(A:'float[:,:]', x:'float[:]', shape_A:'int32[:]'):
    
    import numpy as np

    n = np.int32(shape_A[0])
    m = np.int32(shape_A[1])

    a = np.array(A, order='F')
    y = np.zeros(n)

    alpha = 1.0
    beta  = 0.0

    incx = np.int32(1)
    incy = np.int32(1)
    
    dgemv('N', n, m, alpha, a, n, x, incx, beta, y, incy)
    
