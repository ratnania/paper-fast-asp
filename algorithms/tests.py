#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from numpy import linalg as la

from scipy.sparse.linalg import spsolve_triangular, spsolve
from scipy.sparse.linalg import eigs
from scipy.sparse import identity, diags, csr_matrix, bmat

from utulities import Gauss_Seidel, succesive_over_relaxation, Gauss_Seidel_block_1, Gauss_Seidel_block_2

from tabulate import tabulate



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                   Data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# spsolve   = spsolve_triangular
ks        = [2, 3, 4, 5, 6, 7]
kinds     = ['forward', 'backward', 'symmetric']
m         = 100
methods   = ['GS', 'BGS1', 'BGS2']#, 'BGS']

def create_matrix(k):
    n = 2**k
    A   = np.diag(2.*np.ones(n))\
                   -np.diag(np.ones(n-1), k=1)\
                   -np.diag(np.ones(n-1), k=-1)
   
    A = csr_matrix(A)
    A11 = A12 = A21 = A22 = A                
    
    A = bmat([[A11, A12],
              [A21, A22]])
    
   
    return A, A11, A12, A21, A22
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                    The main function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
def main(method, k, kind, m, spsolve):
    np.random.seed(10)

    tb = time.time()
    # n   = 2**k
    A, A11, A12, A21, A22 = create_matrix(k)

    n = np.shape(A)[0]
    xref = np.random.random(n)
    xref = xref / np.linalg.norm(xref)
    b    = A.dot(xref)
    
    if method == 'BGS1':
        
        # upper_half = np.hsplit(np.vsplit(A, 2)[0], 2)
        # lower_half = np.hsplit(np.vsplit(A, 2)[1], 2)
        
        # A11 = upper_half[0]
        # A12 = upper_half[1]
        # A21 = lower_half[0]
        # A22 = lower_half[1]
        
        
        b1 = np.hsplit(b, 2)[0]
        b2 = np.hsplit(b, 2)[1]
        
        x1, x2 = Gauss_Seidel_block_1(A11               = A11,
                                      A21               = A21,
                                      A22               = A22,
                                      b1                 = b1, 
                                      b2                 = b2, 
                                      kind              = kind,
                                      x0                = None, 
                                      iterations_number = m, 
                                      spsolve           = spsolve)
        
        x = np.concatenate([x1, x2])
        te = time.time()
        
    if method == 'BGS2':
        
        b1 = np.hsplit(b, 2)[0]
        b2 = np.hsplit(b, 2)[1]
        
        x1, x2 = Gauss_Seidel_block_2(A11               = A11,
                                      A12               = A12,
                                      A22               = A22,
                                      b1                 = b1, 
                                      b2                 = b2, 
                                      kind              = kind,
                                      x0                = None, 
                                      iterations_number = m, 
                                      spsolve           = spsolve)
        
        x = np.concatenate([x1, x2])
        te = time.time()
    
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
    
    if method=='GS':
        main_tables(method, ks, kinds, m, spsolve_triangular)
    elif method=='BGS1' or method=='BGS2':
        main_tables(method, ks, kinds, m, spsolve)
    
        
  

        
        
        
        
    
    
    

    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        
    


    

        