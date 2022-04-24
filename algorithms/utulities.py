#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import tril, triu, diags
from scipy.sparse import csr_matrix



def Gauss_Seidel(A, b, kind, x0, iterations_number, spsolve):
    if x0 == None:
        n  = A.shape[0]
        x0 = np.zeros(n)
    A = csr_matrix(A)
    m = iterations_number
    x = x0
    
    if kind == 'forward':
        L = tril(A, format="csc")
        for i in range(m):
            x += spsolve(L, b-A.dot(x), lower=True)
    
    elif kind == 'backward': 
        U = triu(A, format="csr")
        for i in range(m):
            x += spsolve(U, b-A.dot(x), lower=False)
    
    elif kind == 'symmetric': 
        L = tril(A, format="csc")
        U = triu(A, format="csr")
        for i in range(m):
            x += spsolve(L, b-A.dot(x), lower=True)
            x += spsolve(U, b-A.dot(x), lower=False)
    else:
        raise NotImplementedError('kind not available')
    
    return x

def succesive_over_relaxation(A, b, w, kind, x0, iterations_number, spsolve):
    if x0 == None:
        n  = A.shape[0]
        x0 = np.zeros(n)
    A = csr_matrix(A)
    m = iterations_number
    x = x0
    
    if kind == 'forward':
        E = tril(A, -1, format="csc")
        D = diags(A.diagonal(), format="csc")
       
        for i in range(m):
            x += spsolve(D+w*E, w*(b-A.dot(x)), lower=True)
    
    elif kind == 'backward': 
        F = triu(A, format="csr")
        D = diags(A.diagonal(), format="csc")
        for i in range(m):
            x += spsolve(D+w*F, w*(b-A.dot(x)), lower=False)
    
    elif kind == 'symmetric': 
        E = tril(A, -1, format="csc")
        F = triu(A, format="csr")
        D = diags(A.diagonal(), format="csc")
        for i in range(m):
            x += spsolve(D+w*E, w*(b-A.dot(x)), lower=True)
            x += spsolve(D+w*F, w*(b-A.dot(x)), lower=False)
    else:
        raise NotImplementedError('kind not available')
    
    return x
        
        
    





        
        
        
        
    
    
    

    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        

        
    


    

        