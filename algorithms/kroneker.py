  
# def multi_index():
#     pass
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
        
        
        
