"""
implements kron product in the case of CSR matrices
"""

# =========================================================================
def vec_2d(x_mat: 'float[:,:]', x: 'float[:]'):
    """Convert a matrix to a vector form."""

    n1, n2 = x_mat.shape

    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x[i] = x_mat[i1, i2]

# =========================================================================
def unvec_2d(x: 'float[:]', n1: int, n2: int, x_mat: 'float[:,:]'):
    """Convert a vector to a matrix form."""

    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x_mat[i1, i2] = x[i]

# =========================================================================
def vec_3d(x_mat: 'float[:,:,:]', x: 'float[:]'):
    """Convert a matrix to a vector form."""

    n1, n2, n3 = x_mat.shape

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                i = i3 + (i2 + i1 * n2) * n3
                x[i] = x_mat[i1, i2, i3]

# =========================================================================
def unvec_3d(x: 'float[:]', n1: int, n2: int, n3: int, x_mat: 'float[:,:,:]'):
    """Convert a vector to a matrix form."""

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                i = i3 + (i2 + i1 * n2) * n3
                x_mat[i1, i2, i3] = x[i]

# =========================================================================
def mxm(A_data: 'float[:]', A_ind: 'int32[:]', A_ptr: 'int32[:]',
        x: 'float[:,:]', y: 'float[:,:]'):

    """Matrix-Vector product."""

    n = len(A_ptr) - 1
    m = x.shape[0]

    for k in range(m):
        for i in range(n):
            wi = 0.
            for j in range(A_ptr[i], A_ptr[i+1]):
                wi += A_data[j] * x[k,A_ind[j]]
            y[i,k] = wi

# =========================================================================
def unvec_2d_omp(x: 'float[:]', n1: int, n2: int, x_mat: 'float[:,:]'):
    """Convert a vector to a matrix form."""

    #$ omp for schedule(runtime)
    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x_mat[i1, i2] = x[i]

# =========================================================================
def vec_2d_omp(x_mat: 'float[:,:]', x: 'float[:]'):
    """Convert a matrix to a vector form."""

    n1, n2 = x_mat.shape

    #$ omp for schedule(runtime)
    for i1 in range(n1):
        for i2 in range(n2):
            i = i2 + i1 * n2
            x[i] = x_mat[i1, i2]

# =========================================================================
def mxm_omp(A_data: 'float[:]', A_ind: 'int32[:]', A_ptr: 'int32[:]',
        x: 'float[:,:]', y: 'float[:,:]'):

    """Matrix-Vector product."""

    n = len(A_ptr) - 1
    m = x.shape[0]

    #$ omp for schedule(runtime)
    for k in range(m):
        for i in range(n):
            wi = 0.
            for j in range(A_ptr[i], A_ptr[i+1]):
                wi += A_data[j] * x[k,A_ind[j]]
            y[i,k] = wi

# =========================================================================
def kron_2d(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
            A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
            n_rows_1: int, n_cols_1: int,
            n_rows_2: int, n_cols_2: int,
            x: 'float[:]',
            W1: 'float[:,:]', W2: 'float[:,:]',
            y: 'float[:]'):

    # ...
    unvec_2d(x, n_cols_1, n_cols_2, W1[0:n_cols_1, 0:n_cols_2])
    # ...
#    print(W1)
    # ...
    mxm(A2_data, A2_ind, A2_ptr, W1[:n_cols_1,:n_cols_2], W2[:n_rows_2,:n_cols_1])
    # ...

#    print(W2)
    # ...
    mxm(A1_data, A1_ind, A1_ptr, W2[:n_rows_2,:n_cols_1], W1[:n_rows_1,:n_rows_2])
    # ...

    # ...
    vec_2d(W1[0:n_rows_1, 0:n_rows_2], y)
    # ...

# =========================================================================
def kron_3d(A1_data: 'float[:]', A1_ind: 'int32[:]', A1_ptr: 'int32[:]',
            A2_data: 'float[:]', A2_ind: 'int32[:]', A2_ptr: 'int32[:]',
            A3_data: 'float[:]', A3_ind: 'int32[:]', A3_ptr: 'int32[:]',
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

    unvec_2d_omp(x, n_cols_12, n_cols_3, Z1[0:n_cols_12, 0:n_cols_3])

    mxm_omp(A3_data, A3_ind, A3_ptr, Z1[:n_cols_12,:n_cols_3], Z2[:n_rows_3,:n_cols_12])

    #$ omp for schedule(runtime) private(k, Z3, Z4)
    for k in range(n_rows_3):
        kron_2d(A1_data, A1_ind, A1_ptr,
                A2_data, A2_ind, A2_ptr,
                n_rows_1, n_cols_1,
                n_rows_2, n_cols_2,
                Z2[k,0:n_cols_12],
                Z3, Z4,
                Z1[0:n_rows_12,k])

    vec_2d_omp(Z1[0:n_rows_12,0:n_rows_3], y)
    #$ omp end parallel
