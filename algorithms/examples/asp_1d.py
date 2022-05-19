#!/usr/bin/env python
# coding: utf-8

import numpy as np

from numpy import linalg as la

# ========================================================================
def find_index(T,p,t):
    
    assert (t>=0 and t<=1), 't most be in [0,1]'
    
    n=len(T)-p-2
    
    if t==T[n+1]: return n
    
    left = p; right = n+1
    mid  = (left+right)//2

    while (t<T[mid] or t>=T[mid+1]):
        if t<T[mid]: right=mid
        else: left = mid
        mid = (left+right)//2

    return mid

# ========================================================================
def non_zero_basis_functions(T,p,t, index = None):
    
    N=np.ones(p+1)
    
    if index == None:
        i = find_index(T=T,p=p,t=t)
        
    else:
        i = index
        
    for j in range(1,p+1):
        T_      = t-T[i-j+1:i+1]
        _T      = T[i+1:i+j+1]-t
        q       = N[:j]/(_T+T_)
        left    = T_*q
        right   = _T*q
        N[:j+1] = np.insert(left,0,0.)+np.append(right,0.)
    return N

# ========================================================================
def basis_funs_1st_der( knots, degree, x):
    
    span = find_index(knots, degree, x)

    values = non_zero_basis_functions(knots, degree-1, x, index = span)

    ders    = np.empty( degree+1, dtype=float )
    saved   = degree * values[0] / (knots[span+1]-knots[span+1-degree])
    ders[0] = -saved

    for j in range(1,degree):
        temp    = saved
        saved   = degree * values[j] / (knots[span+j+1]-knots[span+j+1-degree])
        ders[j] = temp - saved

    ders[degree] = saved

    return ders

# ========================================================================
def location_matrix(T,p):
    
    m  = len(T)
    n  = m-p-1
    N  = n-p+1
    LM = np.zeros((N-1,p+1),dtype=int)
    
    for e in range(N-1):
        LM[e,:] = range(e,e+p+1)
        
    return LM

# ========================================================================
def Gauss_Legendre(ordergl,tol=10e-14):
    
    # TODO: TO BE REMOVED TO UTILITIES 
    """
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
    """
    m = ordergl + 1
    from math import cos,pi
    from numpy import zeros

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p  = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p1,dp

    A = zeros(m)
    x = zeros(m)
    nRoots = (m + 1)// 2          # Number of non-neg. roots
    
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30):
            p,dp = legendre(t,m)          # Newton-Raphson
            dt   = -p/dp; t = t + dt        # method
            if abs(dt) < tol:
                x[i]     = t; x[m-i-1] = -t
                A[i]     = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x,A


# ========================================================================
def integrate_function(g,a,b,ordergl):
    
    c     = 0.5*(b-a)
    G     = lambda t : g(0.5*((b-a)*t+b+a))
    (r,w) = Gauss_Legendre(ordergl,tol=10e-14)
    s     = np.dot(w,G(r))
    
    return c*s


# ========================================================================
def integrate_matrix_function(g,a,b,ordergl):
    
    c     = 0.5*(b-a)
    G     = lambda t : g(0.5*((b-a)*t+b+a))
    (r,w) = Gauss_Legendre(ordergl,tol=10e-14)
    
    return (c*np.sum(w*G(r).T,axis=2)).T

# ========================================================================
def integrate_vector_function(g,a,b,ordergl):
    
    c     = 0.5*(b-a)
    G     = lambda t : g(0.5*((b-a)*t+b+a))
    (r,w) = Gauss_Legendre(ordergl,tol=10e-14)
    
    return c*np.sum(w*G(r).T,axis=1)

# ========================================================================
def knot_vector(N,p):
    
    # N must mean the number of elements
    N        = N + 1
    n        = N+p-1
    m        = n+p+1
    T        = np.zeros(m)
    T_int    = np.linspace(0,1,N)
    T[p:n+1] = T_int
    T[n+1:]  = np.ones(len(T[n+1:]))
    
    return T

# ========================================================================
def M_matrix(T,p):
    
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    ordergl = p+1
    A       = np.zeros((n,n))

    B = lambda t : np.outer(non_zero_basis_functions(T,p,t),
                            non_zero_basis_functions(T,p,t))
    B = np.vectorize(B,signature='()->(n,n)')

    for e in range(N-1):
        a              = T[p+e]
        b              = T[p+e+1]
        Ae             = integrate_matrix_function(B,a,b,ordergl)
        a1             = location_matrix(T,p)[e,:][0]
        a2             = location_matrix(T,p)[e,:][-1]+1
        A[a1:a2,a1:a2] = A[a1:a2,a1:a2]+Ae
        
    return A[1:,1:][:-1,:-1]

# ========================================================================
def K_matrix(T,p):
    
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    ordergl = p+1
    A       = np.zeros((n,n))

    dB = lambda t : np.outer(basis_funs_1st_der(T,p,t),
                             basis_funs_1st_der(T,p,t))
    dB = np.vectorize(dB,signature='()->(n,n)')

    for e in range(N-1):
        a              = T[p+e]
        b              = T[p+e+1]
        Ae             = integrate_matrix_function(dB,a,b,ordergl)
        a1             = location_matrix(T,p)[e,:][0]
        a2             = location_matrix(T,p)[e,:][-1]+1
        A[a1:a2,a1:a2] = A[a1:a2,a1:a2]+Ae
        
    return A[1:,1:][:-1,:-1]

# ========================================================================
def non_zero_Curry_basis_functions(T,p,t, normalize=True):
    
    j = find_index(T,p,t)
    B = non_zero_basis_functions(T,p-1,t, index = j)
    
    if normalize:
        diff = T[j+1:j+p+1]-T[j-p+1:j+1]
        return (p/diff)*B
    else:
        return B

# ========================================================================
def location_matrix_2(T,p):
    
    m  = len(T)
    n  = m-p-1
    N  = n-p+1
    LM = np.zeros((N-1,p),dtype=int)
    
    for e in range(N-1):
        LM[e,:] = range(e,e+p)
    return LM

# ========================================================================
def D_matrix(T, p, normalize=True):
    
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    ordergl = p+1
    A       = np.zeros((n-1,n-1))

    # ...
    B = lambda t : np.outer(non_zero_Curry_basis_functions(T,p,t, normalize=normalize),
                            non_zero_Curry_basis_functions(T,p,t, normalize=normalize))
    # ...

    B = np.vectorize(B,signature='()->(n,n)')

    for e in range(N-1):
        a              = T[p+e]
        b              = T[p+e+1]
        Ae             = integrate_matrix_function(B,a,b,ordergl)
        a1             = location_matrix_2(T,p)[e,:][0]
        a2             = location_matrix_2(T,p)[e,:][-1]+1
        A[a1:a2,a1:a2] = A[a1:a2,a1:a2]+Ae
        
    return A

# ========================================================================
def R_matrix(T,p, normalize=True):
    
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    ordergl = p+1
    A       = np.zeros((n-1,n))

    B = lambda t : np.outer(non_zero_Curry_basis_functions(T,p,t, normalize=normalize),
                            basis_funs_1st_der(T,p,t))
    B = np.vectorize(B,signature='()->(n,m)')

    for e in range(N-1):
        a              = T[p+e]
        b              = T[p+e+1]
        Ae             = integrate_matrix_function(B,a,b,ordergl)
        a1             = location_matrix_2(T,p)[e,:][0]
        a2             = location_matrix_2(T,p)[e,:][-1]+1
        a3             = location_matrix(T,p)[e,:][0]
        a4             = location_matrix(T,p)[e,:][-1]+1
        A[a1:a2,a3:a4] = A[a1:a2,a3:a4]+Ae
        
    return A[:,1:-1]


# ========================================================================
def Greville_points(T,p):
    
    m = len(T)
    n = m-p-1
    g = np.zeros(n)
    
    for j in range(n):
        g[j] = sum(T[j+1:j+p+1])/float(p)
        
    return g

# ========================================================================
def basis_functions(T,p,t):
    
    m = len(T)
    n = m-p-1
    i = find_index(T,p,t)
    B = np.zeros(n)
    
    B[i-p:i+1] = non_zero_basis_functions(T,p,t)
    
    return B

# ========================================================================
def Curry_basis_functions(T,p,t, normalize=True):
    
    m = len(T)
    n = m-p-1
    i = find_index(T,p,t)
    D = np.zeros(n-1)
    
    D[i-p:i] = non_zero_Curry_basis_functions(T,p,t, normalize=normalize)
    
    return D

# ========================================================================
def B_vector(T,p,a,b):
    
    ordergl = p+1
    B       = lambda t : basis_functions(T,p,t)
    B       = np.vectorize(B,signature='()->(n)')
    be      = integrate_vector_function(B,a,b,ordergl)
    
    return be

# ========================================================================
def D_vector(T,p,a,b, normalize=True):
    
    ordergl = p+1
    D       = lambda t : Curry_basis_functions(T,p,t,normalize=normalize)
    D       = np.vectorize(D,signature='()->(n)')
    de      = integrate_vector_function(D,a,b,ordergl)
    
    return de

# ========================================================================
def H_matrix(T,p, normalize=True):
    
    m = len(T)
    n = m-p-1
    g = Greville_points(T,p)
    B = np.zeros((n-1,n))
    D = np.zeros((n-1,n-1))
    
    for i in range(n-1):
        a      = g[i]
        b      = g[i+1]
        B[i,:] = B_vector(T,p,a,b)
        D[i,:] = D_vector(T,p,a,b,normalize=normalize)

    Q = la.solve(D,B)
    
    return Q[:,1:-1]

# ========================================================================
def collocation_matrix(T, p):
    #TODO IMPLEMENTE THE COLLOCATION MATRIX FOR GENREAL GRID POINTS
    m = len(T)
    n = m-p-1
    g = Greville_points(T,p)
    M = np.zeros((n,n))
    
    for i in range(len(g)):
        j            = find_index(T,p,g[i])
        M[i,j-p:j+1] = non_zero_basis_functions(T,p,g[i])
    return M

# ========================================================================
def histopolation_matrix(T, p, normalize=True):
    
    m = len(T)
    n = m-p-1
    g = Greville_points(T,p)
    M = np.zeros((n-1,n-1))
    
    for i in range(n-1):
        a      = g[i]
        b      = g[i+1]
        M[i,:] = D_vector(T,p,a,b, normalize=normalize)  
        
    return M

        
# ========================================================================
def B_spline_interpolation_coefficients(T, p, u):
    
    g = Greville_points(T, p)
    y = u(g)
    M = collocation_matrix(T, p)
    c = la.solve(M, y)
    
    return c

# ========================================================================
def spline_histopolation_coefficients(T, p, u, normalize=True):
    
    g       = Greville_points(T, p)
    n       = g.shape[0]
    y       = np.empty(n-1, dtype=float)
    ordergl = p+1
    
    for i in range(n-1):
        a    = g[i]
        b    = g[i+1]
        y[i] = integrate_function(u, a, b, ordergl)
        
    M = histopolation_matrix(T, p, normalize = normalize)
    c = la.solve(M, y)
    
    return c

# ========================================================================
def B_spline_interpolation_function(T, p, u, t):
    
    c  = B_spline_interpolation_coefficients(T=T, p=p, u=u)[1:-1]
    b  = basis_functions(T=T,p=p,t=t)[1:-1]
    uh = np.dot(c, b) 
    
    return uh

# ========================================================================
def spline_histopolation_function(T, p, u, t, normalize=True):
    
    c  = spline_histopolation_coefficients(T, p, u, normalize = normalize)
    b  = Curry_basis_functions(T=T,p=p,t=t)
    uh = np.dot(c, b) 
    
    return uh

# ========================================================================
def I_matrix(T,p):
    
    m = len(T)
    n = m-p-1
    
    return np.eye(n-2)

# ========================================================================
def b_vector(T,p,f):
    
    ordergl = p+1
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    F       = np.zeros(n)
    
    B = lambda t : non_zero_basis_functions(T,p,t)*f(t)
    B = np.vectorize(B,signature='()->(n)')
   
    for e in range(N-1):
        a        = T[p+e]
        b        = T[p+e+1]
        be       = integrate_vector_function(B,a,b,ordergl)
        a1       = location_matrix(T,p)[e,:][0]
        a2       = location_matrix(T,p)[e,:][-1]+1
        F[a1:a2] = F[a1:a2]+be
    return F[1:-1]

# ========================================================================
def d_vector(T,p,f,normalize=True):
    
    ordergl = p+1
    m       = len(T)
    n       = m-p-1
    N       = n-p+1
    F       = np.zeros(n-1)
    
    B = lambda t : non_zero_Curry_basis_functions(T,p,t, normalize=normalize)*f(t)
    B = np.vectorize(B,signature='()->(n)')
   
    for e in range(N-1):
        a        = T[p+e]
        b        = T[p+e+1]
        be       = integrate_vector_function(B,a,b,ordergl)
        a1       = location_matrix_2(T,p)[e,:][0]
        a2       = location_matrix_2(T,p)[e,:][-1]+1
        F[a1:a2] = F[a1:a2]+be
    return F



'''
from multiprocessing import Pool, freeze_support

def f_sum(a, b):
    return a + b

def test():
    process_pool = Pool(3)
    data = [(1, 1), (2, 1), (3, 1)]
    output = process_pool.starmap(f_sum, data)
    print(output)
    
if __name__ == '__main__':
    1+1==2
    test()
'''
    