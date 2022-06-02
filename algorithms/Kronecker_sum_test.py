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

ps            = (1, 2, 3,)
ks            = (2, 3, 4,)
tau           = 10.**2
problems      = ['curl']
ms            = [1, 10]



'''
p1 = 3
p2 = 2
p3 = 1
n1 = 2**2
n2 = 2**4
n3 = 2**3

p  = (p1, p2, p3)
n  = (n1, n2, n3)

T1 = knot_vector(n1, p1)
T2 = knot_vector(n2, p2)
T3 = knot_vector(n3, p3)
T  = (T1, T2, T3)

tau = 10.**2
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

c1 = confficients[0] 
c2 = confficients[1] 
c3 = confficients[2] 
 
(D1, M2, K3, D1, K2, M3, D1, M2, M3) = A11
(K1, D2, M3, M1, D2, K3, M1, D2, M3) = A22
(M1, K2, D3, K1, M2, D3, M1, M2, D3) = A33

M11 = sp_kron(sp_kron(D1,M2),K3) + sp_kron(sp_kron(D1,K2),M3) + tau*sp_kron(sp_kron(D1,M2),M3)
M22 = sp_kron(sp_kron(K1,D2),M3)+sp_kron(sp_kron(M1,D2),K3) + tau*sp_kron(sp_kron(M1,D2),M3) 
M33 = sp_kron(sp_kron(M1,K2),D3) + sp_kron(sp_kron(K1,M2),D3) + tau*sp_kron(sp_kron(M1,M2),D3)  
'''

'''
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
'''





'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test1
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A1, A2, A3, B1, B2, B3, C1, C2, C3 = A11

N = np.shape(A1)[0]*np.shape(A2)[0]*np.shape(A3)[0]
b = np.ones(N)

xref = la.solve(tril(M11, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A11, 
                     confficients = confficients[0], 
                     b = b, 
                     lower = True)

print(r'err_lower: %.e' %(max(abs(x-xref))))


xref = la.solve(triu(M11, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A11, 
                     confficients = confficients[0], 
                     b = b, 
                     lower = False)

print(r'err_upper: %.e' %(max(abs(x-xref))))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test2
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A1, A2, A3, B1, B2, B3, C1, C2, C3 = A22

N = np.shape(A1)[0]*np.shape(A2)[0]*np.shape(A3)[0]
b = np.ones(N)

xref = la.solve(tril(M22, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A22, 
                      confficients = confficients[0], 
                      b = b, 
                      lower = True)

print(r'err_lower: %.e' %(max(abs(x-xref))))

xref = la.solve(triu(M22, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A22, 
                     confficients = confficients[0], 
                     b = b, 
                     lower = False)

print(r'err_upper: %.e' %(max(abs(x-xref))))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test3
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A1, A2, A3, B1, B2, B3, C1, C2, C3 = A33

N = np.shape(A1)[0]*np.shape(A2)[0]*np.shape(A3)[0]
b = np.ones(N)

xref = la.solve(tril(M33, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A33, 
                     confficients = confficients[0], 
                     b = b, 
                     lower = True)

print(r'err_lower: %.e' %(max(abs(x-xref))))

xref = la.solve(triu(M33, format='csr').toarray(), b)
x = spsolve_kron_sum(matrices = A33, 
                     confficients = confficients[0], 
                     b = b, 
                     lower = False)

print(r'err_upper: %.e' %(max(abs(x-xref))))
'''

'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test4
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A = [A11, None, None,
     A21, A22, None,
     A31, A32, A33]

A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33


n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0]

b = np.ones(n1+n2+n3)

x = spsolve_kron_block(A, confficients, b)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test5
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A = [A11, A12, A13,
     None, A22, A23,
     None, None, A33]

A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33


n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0]

b = np.ones(n1+n2+n3)

x = spsolve_kron_block(A, confficients, b)
'''

'''
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   test6
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
A = [A11, A12, A13,
     A21, A22, A23,
     A31, A32, A33]

A1_1, A2_1, A3_1, B1_1, B2_1, B3_1, C1_1, C2_1, C3_1 = A11
A1_2, A2_2, A3_2, B1_2, B2_2, B3_2, C1_2, C2_2, C3_2 = A22
A1_3, A2_3, A3_3, B1_3, B2_3, B3_3, C1_3, C2_3, C3_3 = A33


n1 = A1_1.shape[0]*A2_1.shape[0]*A3_1.shape[0]
n2 = A1_2.shape[0]*A2_2.shape[0]*A3_2.shape[0]
n3 = A1_3.shape[0]*A2_3.shape[0]*A3_3.shape[0]

xref = np.ones(n1+n2+n3)
xref = xref/la.norm(xref)
x1 = xref[:n1]
x2 = xref[n1:n1+n2]
x3 = xref[n1+n2:]
 
b1 = M11.dot(x1)+A12.dot(x2)+A13.dot(x3) 
b2 = A21.dot(x1)+M22.dot(x2)+A23.dot(x3) 
b3 = A31.dot(x1)+A32.dot(x2)+M33.dot(x3)
b  = np.concatenate([b1,b2,b3])

#b = np.ones(n1+n2+n3)

x = Gauss_Seidel(A, confficients, b, None,10)


print(r'err: %.e' %(max(abs(x-xref))))
'''



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

def main(k, p, m, problem):
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
    
    (D1, M2, K3, D1, K2, M3, D1, M2, M3) = A11
    (K1, D2, M3, M1, D2, K3, M1, D2, M3) = A22
    (M1, K2, D3, K1, M2, D3, M1, M2, D3) = A33

    M11 = sp_kron(sp_kron(D1,M2),K3) + sp_kron(sp_kron(D1,K2),M3) + tau*sp_kron(sp_kron(D1,M2),M3)
    M22 = sp_kron(sp_kron(K1,D2),M3)+sp_kron(sp_kron(M1,D2),K3) + tau*sp_kron(sp_kron(M1,D2),M3) 
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
    
    b1 = M11.dot(x1)+A12.dot(x2)+A13.dot(x3) 
    b2 = A21.dot(x1)+M22.dot(x2)+A23.dot(x3) 
    b3 = A31.dot(x1)+A32.dot(x2)+M33.dot(x3)
    b  = np.concatenate([b1,b2,b3])
    
    x  = Gauss_Seidel(A, confficients, b, None, m)
    
    te = time.time()
    
    err_2   = la.norm(x-xref)
    err_max = max(abs(x-xref))
    
    info['err_2']        = err_2 
    info['err_max']      = err_max 
    info['elapsed'] = te-tb
    
    print('err_2: ', sprint(err_2)) 
    print('err_max: ', sprint(err_max))
    print('elapsed_time: ', sprint(te-tb))
    
    return info
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates the tables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main_tables(problem):
    
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
    
        write_table(d, m, folder = folder, kind ='err_2')
        write_table(d, m, folder = folder, kind ='err_max')
        write_table(d, m, folder = folder, kind ='elapsed')
            
                   
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Run tests
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# =========================================================================
if __name__ == '__main__':
    for problem in problems:
        main_tables(problem  = problem)
            

        
    
    
    


