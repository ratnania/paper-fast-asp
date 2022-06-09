#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API
#include "numpy_version.h"
#include "numpy/arrayobject.h"
#include "cwrapper.h"
#include <stdlib.h>
#include "ndarrays.h"
#include <stdint.h>
#include "cwrapper_ndarrays.h"


void spsolve_kron_csr_3_sum_lower(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y);
void spsolve_kron_csr_3_sum_upper(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y);

/*........................................*/


/*........................................*/

/*........................................*/
PyObject *spsolve_kron_csr_3_sum_lower_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray A1_data = {.shape = NULL};
    t_ndarray A1_ind = {.shape = NULL};
    t_ndarray A1_ptr = {.shape = NULL};
    t_ndarray A2_data = {.shape = NULL};
    t_ndarray A2_ind = {.shape = NULL};
    t_ndarray A2_ptr = {.shape = NULL};
    t_ndarray A3_data = {.shape = NULL};
    t_ndarray A3_ind = {.shape = NULL};
    t_ndarray A3_ptr = {.shape = NULL};
    t_ndarray B1_data = {.shape = NULL};
    t_ndarray B1_ind = {.shape = NULL};
    t_ndarray B1_ptr = {.shape = NULL};
    t_ndarray B2_data = {.shape = NULL};
    t_ndarray B2_ind = {.shape = NULL};
    t_ndarray B2_ptr = {.shape = NULL};
    t_ndarray B3_data = {.shape = NULL};
    t_ndarray B3_ind = {.shape = NULL};
    t_ndarray B3_ptr = {.shape = NULL};
    t_ndarray C1_data = {.shape = NULL};
    t_ndarray C1_ind = {.shape = NULL};
    t_ndarray C1_ptr = {.shape = NULL};
    t_ndarray C2_data = {.shape = NULL};
    t_ndarray C2_ind = {.shape = NULL};
    t_ndarray C2_ptr = {.shape = NULL};
    t_ndarray C3_data = {.shape = NULL};
    t_ndarray C3_ind = {.shape = NULL};
    t_ndarray C3_ptr = {.shape = NULL};
    double alpha;
    double beta;
    double gamma;
    t_ndarray b = {.shape = NULL};
    t_ndarray y = {.shape = NULL};
    PyArrayObject *A1_data_tmp;
    PyArrayObject *A1_ind_tmp;
    PyArrayObject *A1_ptr_tmp;
    PyArrayObject *A2_data_tmp;
    PyArrayObject *A2_ind_tmp;
    PyArrayObject *A2_ptr_tmp;
    PyArrayObject *A3_data_tmp;
    PyArrayObject *A3_ind_tmp;
    PyArrayObject *A3_ptr_tmp;
    PyArrayObject *B1_data_tmp;
    PyArrayObject *B1_ind_tmp;
    PyArrayObject *B1_ptr_tmp;
    PyArrayObject *B2_data_tmp;
    PyArrayObject *B2_ind_tmp;
    PyArrayObject *B2_ptr_tmp;
    PyArrayObject *B3_data_tmp;
    PyArrayObject *B3_ind_tmp;
    PyArrayObject *B3_ptr_tmp;
    PyArrayObject *C1_data_tmp;
    PyArrayObject *C1_ind_tmp;
    PyArrayObject *C1_ptr_tmp;
    PyArrayObject *C2_data_tmp;
    PyArrayObject *C2_ind_tmp;
    PyArrayObject *C2_ptr_tmp;
    PyArrayObject *C3_data_tmp;
    PyArrayObject *C3_ind_tmp;
    PyArrayObject *C3_ptr_tmp;
    PyObject *alpha_tmp;
    PyObject *beta_tmp;
    PyObject *gamma_tmp;
    PyArrayObject *b_tmp;
    PyArrayObject *y_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "A1_data",
        "A1_ind",
        "A1_ptr",
        "A2_data",
        "A2_ind",
        "A2_ptr",
        "A3_data",
        "A3_ind",
        "A3_ptr",
        "B1_data",
        "B1_ind",
        "B1_ptr",
        "B2_data",
        "B2_ind",
        "B2_ptr",
        "B3_data",
        "B3_ind",
        "B3_ptr",
        "C1_data",
        "C1_ind",
        "C1_ptr",
        "C2_data",
        "C2_ind",
        "C2_ptr",
        "C3_data",
        "C3_ind",
        "C3_ptr",
        "alpha",
        "beta",
        "gamma",
        "b",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!OOOO!O!", kwlist, &PyArray_Type, &A1_data_tmp, &PyArray_Type, &A1_ind_tmp, &PyArray_Type, &A1_ptr_tmp, &PyArray_Type, &A2_data_tmp, &PyArray_Type, &A2_ind_tmp, &PyArray_Type, &A2_ptr_tmp, &PyArray_Type, &A3_data_tmp, &PyArray_Type, &A3_ind_tmp, &PyArray_Type, &A3_ptr_tmp, &PyArray_Type, &B1_data_tmp, &PyArray_Type, &B1_ind_tmp, &PyArray_Type, &B1_ptr_tmp, &PyArray_Type, &B2_data_tmp, &PyArray_Type, &B2_ind_tmp, &PyArray_Type, &B2_ptr_tmp, &PyArray_Type, &B3_data_tmp, &PyArray_Type, &B3_ind_tmp, &PyArray_Type, &B3_ptr_tmp, &PyArray_Type, &C1_data_tmp, &PyArray_Type, &C1_ind_tmp, &PyArray_Type, &C1_ptr_tmp, &PyArray_Type, &C2_data_tmp, &PyArray_Type, &C2_ind_tmp, &PyArray_Type, &C2_ptr_tmp, &PyArray_Type, &C3_data_tmp, &PyArray_Type, &C3_ind_tmp, &PyArray_Type, &C3_ptr_tmp, &alpha_tmp, &beta_tmp, &gamma_tmp, &PyArray_Type, &b_tmp, &PyArray_Type, &y_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(A1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_data = pyarray_to_ndarray(A1_data_tmp);
    }
    if (!pyarray_check(A1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_ind = pyarray_to_ndarray(A1_ind_tmp);
    }
    if (!pyarray_check(A1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_ptr = pyarray_to_ndarray(A1_ptr_tmp);
    }
    if (!pyarray_check(A2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_data = pyarray_to_ndarray(A2_data_tmp);
    }
    if (!pyarray_check(A2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_ind = pyarray_to_ndarray(A2_ind_tmp);
    }
    if (!pyarray_check(A2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_ptr = pyarray_to_ndarray(A2_ptr_tmp);
    }
    if (!pyarray_check(A3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_data = pyarray_to_ndarray(A3_data_tmp);
    }
    if (!pyarray_check(A3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_ind = pyarray_to_ndarray(A3_ind_tmp);
    }
    if (!pyarray_check(A3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_ptr = pyarray_to_ndarray(A3_ptr_tmp);
    }
    if (!pyarray_check(B1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_data = pyarray_to_ndarray(B1_data_tmp);
    }
    if (!pyarray_check(B1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_ind = pyarray_to_ndarray(B1_ind_tmp);
    }
    if (!pyarray_check(B1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_ptr = pyarray_to_ndarray(B1_ptr_tmp);
    }
    if (!pyarray_check(B2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_data = pyarray_to_ndarray(B2_data_tmp);
    }
    if (!pyarray_check(B2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_ind = pyarray_to_ndarray(B2_ind_tmp);
    }
    if (!pyarray_check(B2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_ptr = pyarray_to_ndarray(B2_ptr_tmp);
    }
    if (!pyarray_check(B3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_data = pyarray_to_ndarray(B3_data_tmp);
    }
    if (!pyarray_check(B3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_ind = pyarray_to_ndarray(B3_ind_tmp);
    }
    if (!pyarray_check(B3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_ptr = pyarray_to_ndarray(B3_ptr_tmp);
    }
    if (!pyarray_check(C1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_data = pyarray_to_ndarray(C1_data_tmp);
    }
    if (!pyarray_check(C1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_ind = pyarray_to_ndarray(C1_ind_tmp);
    }
    if (!pyarray_check(C1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_ptr = pyarray_to_ndarray(C1_ptr_tmp);
    }
    if (!pyarray_check(C2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_data = pyarray_to_ndarray(C2_data_tmp);
    }
    if (!pyarray_check(C2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_ind = pyarray_to_ndarray(C2_ind_tmp);
    }
    if (!pyarray_check(C2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_ptr = pyarray_to_ndarray(C2_ptr_tmp);
    }
    if (!pyarray_check(C3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_data = pyarray_to_ndarray(C3_data_tmp);
    }
    if (!pyarray_check(C3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_ind = pyarray_to_ndarray(C3_ind_tmp);
    }
    if (!pyarray_check(C3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_ptr = pyarray_to_ndarray(C3_ptr_tmp);
    }
    if (PyIs_NativeFloat(alpha_tmp))
    {
        alpha = PyDouble_to_Double(alpha_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (PyIs_NativeFloat(beta_tmp))
    {
        beta = PyDouble_to_Double(beta_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (PyIs_NativeFloat(gamma_tmp))
    {
        gamma = PyDouble_to_Double(gamma_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (!pyarray_check(b_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        b = pyarray_to_ndarray(b_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    spsolve_kron_csr_3_sum_lower(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, B1_ind, B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, B3_ptr, C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, C3_data, C3_ind, C3_ptr, alpha, beta, gamma, b, y);
    result = Py_BuildValue("");
    free_pointer(A1_data);
    free_pointer(A1_ind);
    free_pointer(A1_ptr);
    free_pointer(A2_data);
    free_pointer(A2_ind);
    free_pointer(A2_ptr);
    free_pointer(A3_data);
    free_pointer(A3_ind);
    free_pointer(A3_ptr);
    free_pointer(B1_data);
    free_pointer(B1_ind);
    free_pointer(B1_ptr);
    free_pointer(B2_data);
    free_pointer(B2_ind);
    free_pointer(B2_ptr);
    free_pointer(B3_data);
    free_pointer(B3_ind);
    free_pointer(B3_ptr);
    free_pointer(C1_data);
    free_pointer(C1_ind);
    free_pointer(C1_ptr);
    free_pointer(C2_data);
    free_pointer(C2_ind);
    free_pointer(C2_ptr);
    free_pointer(C3_data);
    free_pointer(C3_ind);
    free_pointer(C3_ptr);
    free_pointer(b);
    free_pointer(y);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *spsolve_kron_csr_3_sum_upper_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray A1_data = {.shape = NULL};
    t_ndarray A1_ind = {.shape = NULL};
    t_ndarray A1_ptr = {.shape = NULL};
    t_ndarray A2_data = {.shape = NULL};
    t_ndarray A2_ind = {.shape = NULL};
    t_ndarray A2_ptr = {.shape = NULL};
    t_ndarray A3_data = {.shape = NULL};
    t_ndarray A3_ind = {.shape = NULL};
    t_ndarray A3_ptr = {.shape = NULL};
    t_ndarray B1_data = {.shape = NULL};
    t_ndarray B1_ind = {.shape = NULL};
    t_ndarray B1_ptr = {.shape = NULL};
    t_ndarray B2_data = {.shape = NULL};
    t_ndarray B2_ind = {.shape = NULL};
    t_ndarray B2_ptr = {.shape = NULL};
    t_ndarray B3_data = {.shape = NULL};
    t_ndarray B3_ind = {.shape = NULL};
    t_ndarray B3_ptr = {.shape = NULL};
    t_ndarray C1_data = {.shape = NULL};
    t_ndarray C1_ind = {.shape = NULL};
    t_ndarray C1_ptr = {.shape = NULL};
    t_ndarray C2_data = {.shape = NULL};
    t_ndarray C2_ind = {.shape = NULL};
    t_ndarray C2_ptr = {.shape = NULL};
    t_ndarray C3_data = {.shape = NULL};
    t_ndarray C3_ind = {.shape = NULL};
    t_ndarray C3_ptr = {.shape = NULL};
    double alpha;
    double beta;
    double gamma;
    t_ndarray b = {.shape = NULL};
    t_ndarray y = {.shape = NULL};
    PyArrayObject *A1_data_tmp;
    PyArrayObject *A1_ind_tmp;
    PyArrayObject *A1_ptr_tmp;
    PyArrayObject *A2_data_tmp;
    PyArrayObject *A2_ind_tmp;
    PyArrayObject *A2_ptr_tmp;
    PyArrayObject *A3_data_tmp;
    PyArrayObject *A3_ind_tmp;
    PyArrayObject *A3_ptr_tmp;
    PyArrayObject *B1_data_tmp;
    PyArrayObject *B1_ind_tmp;
    PyArrayObject *B1_ptr_tmp;
    PyArrayObject *B2_data_tmp;
    PyArrayObject *B2_ind_tmp;
    PyArrayObject *B2_ptr_tmp;
    PyArrayObject *B3_data_tmp;
    PyArrayObject *B3_ind_tmp;
    PyArrayObject *B3_ptr_tmp;
    PyArrayObject *C1_data_tmp;
    PyArrayObject *C1_ind_tmp;
    PyArrayObject *C1_ptr_tmp;
    PyArrayObject *C2_data_tmp;
    PyArrayObject *C2_ind_tmp;
    PyArrayObject *C2_ptr_tmp;
    PyArrayObject *C3_data_tmp;
    PyArrayObject *C3_ind_tmp;
    PyArrayObject *C3_ptr_tmp;
    PyObject *alpha_tmp;
    PyObject *beta_tmp;
    PyObject *gamma_tmp;
    PyArrayObject *b_tmp;
    PyArrayObject *y_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "A1_data",
        "A1_ind",
        "A1_ptr",
        "A2_data",
        "A2_ind",
        "A2_ptr",
        "A3_data",
        "A3_ind",
        "A3_ptr",
        "B1_data",
        "B1_ind",
        "B1_ptr",
        "B2_data",
        "B2_ind",
        "B2_ptr",
        "B3_data",
        "B3_ind",
        "B3_ptr",
        "C1_data",
        "C1_ind",
        "C1_ptr",
        "C2_data",
        "C2_ind",
        "C2_ptr",
        "C3_data",
        "C3_ind",
        "C3_ptr",
        "alpha",
        "beta",
        "gamma",
        "b",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!OOOO!O!", kwlist, &PyArray_Type, &A1_data_tmp, &PyArray_Type, &A1_ind_tmp, &PyArray_Type, &A1_ptr_tmp, &PyArray_Type, &A2_data_tmp, &PyArray_Type, &A2_ind_tmp, &PyArray_Type, &A2_ptr_tmp, &PyArray_Type, &A3_data_tmp, &PyArray_Type, &A3_ind_tmp, &PyArray_Type, &A3_ptr_tmp, &PyArray_Type, &B1_data_tmp, &PyArray_Type, &B1_ind_tmp, &PyArray_Type, &B1_ptr_tmp, &PyArray_Type, &B2_data_tmp, &PyArray_Type, &B2_ind_tmp, &PyArray_Type, &B2_ptr_tmp, &PyArray_Type, &B3_data_tmp, &PyArray_Type, &B3_ind_tmp, &PyArray_Type, &B3_ptr_tmp, &PyArray_Type, &C1_data_tmp, &PyArray_Type, &C1_ind_tmp, &PyArray_Type, &C1_ptr_tmp, &PyArray_Type, &C2_data_tmp, &PyArray_Type, &C2_ind_tmp, &PyArray_Type, &C2_ptr_tmp, &PyArray_Type, &C3_data_tmp, &PyArray_Type, &C3_ind_tmp, &PyArray_Type, &C3_ptr_tmp, &alpha_tmp, &beta_tmp, &gamma_tmp, &PyArray_Type, &b_tmp, &PyArray_Type, &y_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(A1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_data = pyarray_to_ndarray(A1_data_tmp);
    }
    if (!pyarray_check(A1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_ind = pyarray_to_ndarray(A1_ind_tmp);
    }
    if (!pyarray_check(A1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A1_ptr = pyarray_to_ndarray(A1_ptr_tmp);
    }
    if (!pyarray_check(A2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_data = pyarray_to_ndarray(A2_data_tmp);
    }
    if (!pyarray_check(A2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_ind = pyarray_to_ndarray(A2_ind_tmp);
    }
    if (!pyarray_check(A2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A2_ptr = pyarray_to_ndarray(A2_ptr_tmp);
    }
    if (!pyarray_check(A3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_data = pyarray_to_ndarray(A3_data_tmp);
    }
    if (!pyarray_check(A3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_ind = pyarray_to_ndarray(A3_ind_tmp);
    }
    if (!pyarray_check(A3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A3_ptr = pyarray_to_ndarray(A3_ptr_tmp);
    }
    if (!pyarray_check(B1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_data = pyarray_to_ndarray(B1_data_tmp);
    }
    if (!pyarray_check(B1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_ind = pyarray_to_ndarray(B1_ind_tmp);
    }
    if (!pyarray_check(B1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B1_ptr = pyarray_to_ndarray(B1_ptr_tmp);
    }
    if (!pyarray_check(B2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_data = pyarray_to_ndarray(B2_data_tmp);
    }
    if (!pyarray_check(B2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_ind = pyarray_to_ndarray(B2_ind_tmp);
    }
    if (!pyarray_check(B2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B2_ptr = pyarray_to_ndarray(B2_ptr_tmp);
    }
    if (!pyarray_check(B3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_data = pyarray_to_ndarray(B3_data_tmp);
    }
    if (!pyarray_check(B3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_ind = pyarray_to_ndarray(B3_ind_tmp);
    }
    if (!pyarray_check(B3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        B3_ptr = pyarray_to_ndarray(B3_ptr_tmp);
    }
    if (!pyarray_check(C1_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_data = pyarray_to_ndarray(C1_data_tmp);
    }
    if (!pyarray_check(C1_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_ind = pyarray_to_ndarray(C1_ind_tmp);
    }
    if (!pyarray_check(C1_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C1_ptr = pyarray_to_ndarray(C1_ptr_tmp);
    }
    if (!pyarray_check(C2_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_data = pyarray_to_ndarray(C2_data_tmp);
    }
    if (!pyarray_check(C2_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_ind = pyarray_to_ndarray(C2_ind_tmp);
    }
    if (!pyarray_check(C2_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C2_ptr = pyarray_to_ndarray(C2_ptr_tmp);
    }
    if (!pyarray_check(C3_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_data = pyarray_to_ndarray(C3_data_tmp);
    }
    if (!pyarray_check(C3_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_ind = pyarray_to_ndarray(C3_ind_tmp);
    }
    if (!pyarray_check(C3_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        C3_ptr = pyarray_to_ndarray(C3_ptr_tmp);
    }
    if (PyIs_NativeFloat(alpha_tmp))
    {
        alpha = PyDouble_to_Double(alpha_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (PyIs_NativeFloat(beta_tmp))
    {
        beta = PyDouble_to_Double(beta_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (PyIs_NativeFloat(gamma_tmp))
    {
        gamma = PyDouble_to_Double(gamma_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native float\"");
        return NULL;
    }
    if (!pyarray_check(b_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        b = pyarray_to_ndarray(b_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    spsolve_kron_csr_3_sum_upper(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, B1_ind, B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, B3_ptr, C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, C3_data, C3_ind, C3_ptr, alpha, beta, gamma, b, y);
    result = Py_BuildValue("");
    free_pointer(A1_data);
    free_pointer(A1_ind);
    free_pointer(A1_ptr);
    free_pointer(A2_data);
    free_pointer(A2_ind);
    free_pointer(A2_ptr);
    free_pointer(A3_data);
    free_pointer(A3_ind);
    free_pointer(A3_ptr);
    free_pointer(B1_data);
    free_pointer(B1_ind);
    free_pointer(B1_ptr);
    free_pointer(B2_data);
    free_pointer(B2_ind);
    free_pointer(B2_ptr);
    free_pointer(B3_data);
    free_pointer(B3_ind);
    free_pointer(B3_ptr);
    free_pointer(C1_data);
    free_pointer(C1_ind);
    free_pointer(C1_ptr);
    free_pointer(C2_data);
    free_pointer(C2_ind);
    free_pointer(C2_ptr);
    free_pointer(C3_data);
    free_pointer(C3_ind);
    free_pointer(C3_ptr);
    free_pointer(b);
    free_pointer(y);
    return result;
}
/*........................................*/

static int exec_func(PyObject* m)
{
    return 0;
}

/*........................................*/

static PyMethodDef kroneker_methods[] = {
    {
        "spsolve_kron_csr_3_sum_lower",
        (PyCFunction)spsolve_kron_csr_3_sum_lower_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    {
        "spsolve_kron_csr_3_sum_upper",
        (PyCFunction)spsolve_kron_csr_3_sum_upper_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};

/*........................................*/

static PyModuleDef_Slot kroneker_slots[] = {
    {Py_mod_exec, exec_func},
    {0, NULL},
};

/*........................................*/

static struct PyModuleDef kroneker_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "kroneker",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    0,
    kroneker_methods,
    kroneker_slots
};

/*........................................*/

PyMODINIT_FUNC PyInit_kroneker(void)
{
    import_array();
    return PyModuleDef_Init(&kroneker_module);
}
