#include "kroneker.h"
#include <stdlib.h>
#include "ndarrays.h"
#include <stdint.h>
#include <math.h>


/*........................................*/
void spsolve_kron_csr_3_sum_lower(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y)
{
    int64_t n1;
    int64_t n2;
    int64_t n3;
    int64_t n;
    int64_t i;
    int64_t i1;
    int64_t r;
    int64_t i2;
    int64_t i3;
    int32_t k1_b;
    int32_t k1_e;
    int32_t k2_b;
    int32_t k2_e;
    int32_t k3_b;
    int32_t k3_e;
    double yi;
    double a_d;
    int64_t k1;
    int32_t j1;
    double a1;
    int64_t k2;
    int32_t j2;
    double a2;
    int64_t k3;
    int32_t j3;
    double a3;
    int64_t j;
    double zi;
    double b_d;
    double wi;
    double c_d;
    n1 = A1_ptr.shape[0] - 1;
    n2 = A2_ptr.shape[0] - 1;
    n3 = A3_ptr.shape[0] - 1;
    n = n1 * n2 * n3;
    for (i = 0; i < n; i += 1)
    {
        i1 = (int64_t)floor((double)(i) / (double)((n2 * n3)));
        r = i - i1 * n2 * n3;
        i2 = (int64_t)floor((double)(r) / (double)(n3));
        i3 = r - i2 * n3;
        /*...*/
        k1_b = GET_ELEMENT(A1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(A1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(A2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(A2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(A3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(A3_ptr, nd_int32, (int64_t)i3 + 1);
        yi = 0.0;
        a_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(A1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(A1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(A2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(A2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(A3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(A3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j < i)
                    {
                        yi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    else if (i == j)
                    {
                        a_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        /*...*/
        k1_b = GET_ELEMENT(B1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(B1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(B2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(B2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(B3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(B3_ptr, nd_int32, (int64_t)i3 + 1);
        zi = 0.0;
        b_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(B1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(B1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(B2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(B2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(B3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(B3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j < i)
                    {
                        zi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    else if (i == j)
                    {
                        b_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        /*...*/
        k1_b = GET_ELEMENT(C1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(C1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(C2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(C2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(C3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(C3_ptr, nd_int32, (int64_t)i3 + 1);
        wi = 0.0;
        c_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(C1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(C1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(C2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(C2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(C3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(C3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j < i)
                    {
                        wi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    else if (i == j)
                    {
                        c_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        GET_ELEMENT(y, nd_double, (int64_t)i) = (GET_ELEMENT(b, nd_double, (int64_t)i) - alpha * yi - beta * zi - gamma * wi) / (alpha * a_d + beta * b_d + gamma * c_d);
    }
}
/*........................................*/
/*........................................*/
void spsolve_kron_csr_3_sum_upper(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y)
{
    int64_t n1;
    int64_t n2;
    int64_t n3;
    int64_t n;
    int64_t i;
    int64_t i1;
    int64_t r;
    int64_t i2;
    int64_t i3;
    int32_t k1_b;
    int32_t k1_e;
    int32_t k2_b;
    int32_t k2_e;
    int32_t k3_b;
    int32_t k3_e;
    double yi;
    double a_d;
    int64_t k1;
    int32_t j1;
    double a1;
    int64_t k2;
    int32_t j2;
    double a2;
    int64_t k3;
    int32_t j3;
    double a3;
    int64_t j;
    double zi;
    double b_d;
    double wi;
    double c_d;
    n1 = A1_ptr.shape[0] - 1;
    n2 = A2_ptr.shape[0] - 1;
    n3 = A3_ptr.shape[0] - 1;
    n = n1 * n2 * n3;
    for (i = n - 1; i > -1; i += -1)
    {
        i1 = (int64_t)floor((double)(i) / (double)((n2 * n3)));
        r = i - i1 * n2 * n3;
        i2 = (int64_t)floor((double)(r) / (double)(n3));
        i3 = r - i2 * n3;
        /*...*/
        k1_b = GET_ELEMENT(A1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(A1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(A2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(A2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(A3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(A3_ptr, nd_int32, (int64_t)i3 + 1);
        yi = 0.0;
        a_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(A1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(A1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(A2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(A2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(A3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(A3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j >= i)
                    {
                        yi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    if (i == j)
                    {
                        a_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        /*...*/
        k1_b = GET_ELEMENT(B1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(B1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(B2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(B2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(B3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(B3_ptr, nd_int32, (int64_t)i3 + 1);
        zi = 0.0;
        b_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(B1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(B1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(B2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(B2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(B3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(B3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j >= i)
                    {
                        zi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    if (i == j)
                    {
                        b_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        /*...*/
        k1_b = GET_ELEMENT(C1_ptr, nd_int32, (int64_t)i1);
        k1_e = GET_ELEMENT(C1_ptr, nd_int32, (int64_t)i1 + 1);
        k2_b = GET_ELEMENT(C2_ptr, nd_int32, (int64_t)i2);
        k2_e = GET_ELEMENT(C2_ptr, nd_int32, (int64_t)i2 + 1);
        k3_b = GET_ELEMENT(C3_ptr, nd_int32, (int64_t)i3);
        k3_e = GET_ELEMENT(C3_ptr, nd_int32, (int64_t)i3 + 1);
        wi = 0.0;
        c_d = 1.0;
        for (k1 = k1_b; k1 < k1_e; k1 += 1)
        {
            j1 = GET_ELEMENT(C1_ind, nd_int32, (int64_t)k1);
            a1 = GET_ELEMENT(C1_data, nd_double, (int64_t)k1);
            for (k2 = k2_b; k2 < k2_e; k2 += 1)
            {
                j2 = GET_ELEMENT(C2_ind, nd_int32, (int64_t)k2);
                a2 = GET_ELEMENT(C2_data, nd_double, (int64_t)k2);
                for (k3 = k3_b; k3 < k3_e; k3 += 1)
                {
                    j3 = GET_ELEMENT(C3_ind, nd_int32, (int64_t)k3);
                    a3 = GET_ELEMENT(C3_data, nd_double, (int64_t)k3);
                    j = j3 + (j2 + j1 * n2) * n3;
                    if (j >= i)
                    {
                        wi += a1 * a2 * a3 * GET_ELEMENT(y, nd_double, j);
                    }
                    if (i == j)
                    {
                        c_d = a1 * a2 * a3;
                    }
                }
            }
        }
        /*...*/
        GET_ELEMENT(y, nd_double, (int64_t)i) = (GET_ELEMENT(b, nd_double, (int64_t)i) - alpha * yi - beta * zi - gamma * wi) / (alpha * a_d + beta * b_d + gamma * c_d);
    }
}
/*........................................*/

