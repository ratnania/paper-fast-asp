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