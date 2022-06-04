#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import os

from numpy import linalg as la

#from scipy.sparse.linalg import spsolve_triangular, spsolve
#from scipy.sparse.linalg import 
#from scipy.sparse import identity, diags, csr_matrix, bmat, tril, triu
from scipy.sparse import kron as sp_kron # TODO remove


from tabulate import tabulate

from examples.matrices_3d import curl_matrix
from examples.asp_1d import knot_vector

from kron_spsolve import Gauss_Seidel

sprint = lambda x: '{:.2e}'.format(x)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   Data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ps            = (1, 2, 3, 4, 5)
ks            = (3, 4, 5, 6)
tau           = 10.**4
problems      = ['curl']
ms            = [1]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates files which contains results
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def mkdir_p(dir):
    # type: (unicode) -> None
    if os.path.isdir(dir):
        return
    os.makedirs(dir)
    
def create_folder(problem):
    
    folder = '3d_results/{problem}'.format(problem=problem)
            
    mkdir_p(os.path.join(folder, 'txt'))
    mkdir_p(os.path.join(folder, 'tex'))
    
    return folder

def write_table(d, m, folder, kind):
    headers = ['grid/degree p']
    
    for p in ps:
        headers.append(str(p))
    
    # add table rows
    rows = []
    for k in ks:
        ncell = str(2**k)
        row = ['$' + ncell + ' \\times ' + ncell + ' \\times ' + ncell + '$']
        for p in ps:
            value = d[p, k][kind]
            if isinstance(value, int):
                v = '$'+str(value) +'$'
            else:
                v =  '$'+sprint(value)+'$' 
            row.append(v)
        rows.append(row)
    
    table = tabulate(rows, headers=headers)
    
    
    
    
    fname = '{label}_m{m}.txt'.format(label=kind, m=m)
    fname = os.path.join('txt', fname)
    fname = os.path.join(folder, fname)
    
    with open(fname, 'w') as f:
        table = tabulate(rows, headers=headers, tablefmt ='fancy_grid')
        f.write(str(table))
    
    fname = '{label}_m{m}.tex'.format(label=kind, m=m)
    fname = os.path.join('tex', fname)
    fname = os.path.join(folder, fname)
    
    with open(fname, 'w') as f:
        table = tabulate(rows, headers=headers, tablefmt='latex_raw')
        f.write(str(table))
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    The main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(k, p, m, problem, err=False):
    print('============pb={problem}, m={m}, p = {p}, k = {k}============'
          .format(problem= problem, m=m, p=p,k=k))
    
    tb = time.time()
    
    n  = 2**k 
    T1 = knot_vector(n, p)
    T2 = knot_vector(n, p)
    T3 = knot_vector(n, p)
    T  = (T1,T2,T3)
    
    if  problem == 'curl':
        matrices, confficients = curl_matrix(T         = T, 
                                             p         = (p, p, p), 
                                             tau       = tau,
                                             normalize = True, 
                                             form      = 'csr')
    else : 
        "TODO"
    
    info = {}
    
    A11 = matrices[0] 
    A12 = matrices[1] 
    A13 = matrices[2] 
    A21 = matrices[3] 
    A22 = matrices[4] 
    A23 = matrices[5] 
    A31 = matrices[6] 
    A32 = matrices[7] 
    A33 = matrices[8] 
    
    A = [A11, A12, A13,
         A21, A22, A23,
         A31, A32, A33]
    
    if err == True:
        (D1, M2, K3, D1, K2, M3, D1, M2, M3) = A11
        (R1, R2_T, M3)                       = A12
        (R1, M2, R3_T)                       = A13
        (R1_T, R2, M3_T)                     = A21
        (K1, D2, M3, M1, D2, K3, M1, D2, M3) = A22
        (M1, R2, R3_T)                       = A23
        (R1_T, M2, R3)                       = A31
        (M1, R2_T, R3)                       = A32
        (M1, K2, D3, K1, M2, D3, M1, M2, D3) = A33
    
        M11 = sp_kron(sp_kron(D1,M2),K3) + sp_kron(sp_kron(D1,K2),M3) + tau*sp_kron(sp_kron(D1,M2),M3)
        M12 = sp_kron(sp_kron(R1,R2_T),M3)
        M13 = sp_kron(sp_kron(R1,M2),R3_T)
        M21 = sp_kron(sp_kron(R1_T,R2),M3_T)
        M22 = sp_kron(sp_kron(K1,D2),M3)+sp_kron(sp_kron(M1,D2),K3) + tau*sp_kron(sp_kron(M1,D2),M3) 
        M23 = sp_kron(sp_kron(M1,R2),R3_T)
        M31 = sp_kron(sp_kron(R1_T,M2),R3)
        M32 = sp_kron(sp_kron(M1,R2_T),R3)
        M33 = sp_kron(sp_kron(M1,K2),D3) + sp_kron(sp_kron(K1,M2),D3) + tau*sp_kron(sp_kron(M1,M2),D3)

        A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
        A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
        A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33
    
        n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
        n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
        n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0] 

        xref = np.random.random(n1+n2+n3)
        xref = xref / np.linalg.norm(xref)
    
        x1 = xref[:n1]
        x2 = xref[n1:n1+n2]
        x3 = xref[n1+n2:]
    
        b1 = M11.dot(x1)+M12.dot(x2)+M13.dot(x3) 
        b2 = M21.dot(x1)+M22.dot(x2)+M23.dot(x3) 
        b3 = M31.dot(x1)+M32.dot(x2)+M33.dot(x3)
        b  = np.concatenate([b1,b2,b3])
    
        x  = Gauss_Seidel(A, confficients, b, None, m)
    
        te = time.time()
    
        err_2   = la.norm(x-xref)
        err_max = max(abs(x-xref))
    
        info['err_2']        = err_2 
        info['err_max']      = err_max 
        info['elapsed']      = te-tb
    
        print('err_2: ', sprint(err_2)) 
        print('err_max: ', sprint(err_max))
        print('elapsed_time: ', sprint(te-tb))
    else:
        n1 = A11[0].shape[0]*A11[1].shape[0]*A11[2].shape[0]
        n2 = A22[0].shape[0]*A22[1].shape[0]*A22[2].shape[0]
        n3 = A33[0].shape[0]*A33[1].shape[0]*A33[2].shape[0]
        
        b = np.random.random(n1+n2+n3)
        
        
        x  = Gauss_Seidel(A, confficients, b, None, m)
        
        te = time.time()
        
        info['elapsed'] = te-tb
        
        print('elapsed_time: ', sprint(te-tb))
        
        
    
    return info
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates the tables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main_tables(problem, err=False):
    
    folder = create_folder(problem  = problem)
    for m in ms:
        d = {}
        for p in ps:
            for k in ks:
                info = main(k        = k, 
                            p        = p, 
                            m        = m,
                            problem  = problem)

                d[p,k] = info
        if err == True:
            write_table(d, m, folder = folder, kind ='err_2')
            write_table(d, m, folder = folder, kind ='err_max')
            write_table(d, m, folder = folder, kind ='elapsed')
        else:
            write_table(d, m, folder = folder, kind ='elapsed')
            
            
                   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Run tests
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# =========================================================================
if __name__ == '__main__':
    for problem in problems:
        main_tables(problem  = problem)
            

        
    
    
    


