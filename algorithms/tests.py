#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from numpy import linalg as la

from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse.linalg import eigs
from scipy.sparse import identity, diags

from utulities import Gauss_Seidel, succesive_over_relaxation

from tabulate import tabulate



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   Data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
spsolve   = spsolve_triangular
ks        = [2, 3, 4, 5, 6, 7]
kinds     = ['forward', 'backward', 'symmetric']
m         = 100
methods   = ['GS', 'SOR']

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    The main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
def main(method, k, kind, m, spsolve):
    tb = time.time()
    n   = 2**k
    A   = np.diag(2.*np.ones(n))\
            -np.diag(np.ones(n-1), k=1)\
                -np.diag(np.ones(n-1), k=-1)
    
    
        
    xref = np.random.random(n)
    xref = xref / np.linalg.norm(xref)
    b    = A.dot(xref)
    
    if method == 'GS':
        x = Gauss_Seidel(A                 = A, 
                         b                 = b, 
                         kind              = kind,
                         x0                = None, 
                         iterations_number = m, 
                         spsolve           = spsolve)
        te = time.time()
    elif method == 'SOR':
        I = identity(n, format="csr")
        D_inv = diags(1./A.diagonal(), format="csc")
        J = I-D_inv*A
        r = max(abs(eigs(J)[0]))
        w = 1+(r/(1+np.sqrt(1-r**2)))**2
        x = succesive_over_relaxation(A                 = A, 
                                      b                 = b, 
                                      w                 = w,
                                      kind              = kind,
                                      x0                = None, 
                                      iterations_number = m, 
                                      spsolve           = spsolve)
        te = time.time()
        
    

    err          = la.norm(x-xref)
    elapsed_time = te-tb
    
    info = {'err': err, 'elapsed_time': elapsed_time}
    
    return info


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates files which contains results
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def write_table(ks, kinds, value):
    headers = ['']
    headers.append('matrix_dim')
    for k in ks:
        n = 2**k
        headers.append(str(n))
        
    rows = []
    
    for kind in kinds:
        row = [kind]
        row.append("\n".join(['elapsed time', 'err']))
        
        for k in ks:
            n = 2**k
            err          = value[kind, k]['err']
            elapsed_time = value[kind, k]['elapsed_time']
            row.append("\n".join(['{:.2e}'.format(elapsed_time), '{:.2e}'.format(err)]))
        rows.append(row)
        
    print(tabulate(rows, headers, "fancy_grid"))
            
            
        
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    Creates the tables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
def main_tables(method, ks, kinds, m, spsolve):
    value = {}
    for kind in kinds:
        for k in ks:
            info = main(method = method, k = k, kind = kind, m=m, spsolve=spsolve)
            value[kind, k] = info
    write_table(ks=ks, kinds=kinds, value=value)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Run tests
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for method in methods:
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('----------------------------------------method={method}---------------------------------------'.format(method=method))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    main_tables(method, ks, kinds, m, spsolve)
    
    
        
  

        
        
        
        
    
    
    

    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        
    


    

        