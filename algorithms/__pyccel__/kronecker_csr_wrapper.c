#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API
#include "numpy_version.h"
#include "numpy/arrayobject.h"
#include "cwrapper.h"
#include <stdlib.h>
#include "ndarrays.h"
#include "cwrapper_ndarrays.h"
#include <stdint.h>


void bind_c_vec_2d(int64_t, int64_t, double*, int64_t, double*);
void bind_c_unvec_2d(int64_t, double*, int64_t, int64_t, int64_t, int64_t, double*);
void bind_c_vec_3d(int64_t, int64_t, int64_t, double*, int64_t, double*);
void bind_c_unvec_3d(int64_t, double*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double*);
void bind_c_mxm(int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, int64_t, double*, int64_t, int64_t, double*);
void bind_c_unvec_2d_omp(int64_t, double*, int64_t, int64_t, int64_t, int64_t, double*);
void bind_c_vec_2d_omp(int64_t, int64_t, double*, int64_t, double*);
void bind_c_mxm_omp(int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, int64_t, double*, int64_t, int64_t, double*);
void bind_c_kron_2d(int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, double*);
void bind_c_kron_3d(int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, double*, int64_t, int32_t*, int64_t, int32_t*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, int64_t, double*, int64_t, double*);

/*........................................*/


/*........................................*/

/*........................................*/
PyObject *vec_2d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x_mat = {.shape = NULL};
    t_ndarray x = {.shape = NULL};
    PyArrayObject *x_mat_tmp;
    PyArrayObject *x_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x_mat",
        "x",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyArray_Type, &x_mat_tmp, &PyArray_Type, &x_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    bind_c_vec_2d(nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_data(&x_mat), nd_ndim(&x, 0), nd_data(&x));
    result = Py_BuildValue("");
    free_pointer(x_mat);
    free_pointer(x);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *unvec_2d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x = {.shape = NULL};
    int64_t n1;
    int64_t n2;
    t_ndarray x_mat = {.shape = NULL};
    PyArrayObject *x_tmp;
    PyObject *n1_tmp;
    PyObject *n2_tmp;
    PyArrayObject *x_mat_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x",
        "n1",
        "n2",
        "x_mat",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OOO!", kwlist, &PyArray_Type, &x_tmp, &n1_tmp, &n2_tmp, &PyArray_Type, &x_mat_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (PyIs_NativeInt(n1_tmp))
    {
        n1 = PyInt64_to_Int64(n1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n2_tmp))
    {
        n2 = PyInt64_to_Int64(n2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    bind_c_unvec_2d(nd_ndim(&x, 0), nd_data(&x), n1, n2, nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_data(&x_mat));
    result = Py_BuildValue("");
    free_pointer(x);
    free_pointer(x_mat);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *vec_3d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x_mat = {.shape = NULL};
    t_ndarray x = {.shape = NULL};
    PyArrayObject *x_mat_tmp;
    PyArrayObject *x_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x_mat",
        "x",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyArray_Type, &x_mat_tmp, &PyArray_Type, &x_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 3, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    bind_c_vec_3d(nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_ndim(&x_mat, 2), nd_data(&x_mat), nd_ndim(&x, 0), nd_data(&x));
    result = Py_BuildValue("");
    free_pointer(x_mat);
    free_pointer(x);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *unvec_3d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x = {.shape = NULL};
    int64_t n1;
    int64_t n2;
    int64_t n3;
    t_ndarray x_mat = {.shape = NULL};
    PyArrayObject *x_tmp;
    PyObject *n1_tmp;
    PyObject *n2_tmp;
    PyObject *n3_tmp;
    PyArrayObject *x_mat_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x",
        "n1",
        "n2",
        "n3",
        "x_mat",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OOOO!", kwlist, &PyArray_Type, &x_tmp, &n1_tmp, &n2_tmp, &n3_tmp, &PyArray_Type, &x_mat_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (PyIs_NativeInt(n1_tmp))
    {
        n1 = PyInt64_to_Int64(n1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n2_tmp))
    {
        n2 = PyInt64_to_Int64(n2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n3_tmp))
    {
        n3 = PyInt64_to_Int64(n3_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 3, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    bind_c_unvec_3d(nd_ndim(&x, 0), nd_data(&x), n1, n2, n3, nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_ndim(&x_mat, 2), nd_data(&x_mat));
    result = Py_BuildValue("");
    free_pointer(x);
    free_pointer(x_mat);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *mxm_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray A_data = {.shape = NULL};
    t_ndarray A_ind = {.shape = NULL};
    t_ndarray A_ptr = {.shape = NULL};
    t_ndarray x = {.shape = NULL};
    t_ndarray y = {.shape = NULL};
    PyArrayObject *A_data_tmp;
    PyArrayObject *A_ind_tmp;
    PyArrayObject *A_ptr_tmp;
    PyArrayObject *x_tmp;
    PyArrayObject *y_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "A_data",
        "A_ind",
        "A_ptr",
        "x",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!", kwlist, &PyArray_Type, &A_data_tmp, &PyArray_Type, &A_ind_tmp, &PyArray_Type, &A_ptr_tmp, &PyArray_Type, &x_tmp, &PyArray_Type, &y_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(A_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_data = pyarray_to_ndarray(A_data_tmp);
    }
    if (!pyarray_check(A_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_ind = pyarray_to_ndarray(A_ind_tmp);
    }
    if (!pyarray_check(A_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_ptr = pyarray_to_ndarray(A_ptr_tmp);
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    bind_c_mxm(nd_ndim(&A_data, 0), nd_data(&A_data), nd_ndim(&A_ind, 0), nd_data(&A_ind), nd_ndim(&A_ptr, 0), nd_data(&A_ptr), nd_ndim(&x, 0), nd_ndim(&x, 1), nd_data(&x), nd_ndim(&y, 0), nd_ndim(&y, 1), nd_data(&y));
    result = Py_BuildValue("");
    free_pointer(A_data);
    free_pointer(A_ind);
    free_pointer(A_ptr);
    free_pointer(x);
    free_pointer(y);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *unvec_2d_omp_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x = {.shape = NULL};
    int64_t n1;
    int64_t n2;
    t_ndarray x_mat = {.shape = NULL};
    PyArrayObject *x_tmp;
    PyObject *n1_tmp;
    PyObject *n2_tmp;
    PyArrayObject *x_mat_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x",
        "n1",
        "n2",
        "x_mat",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!OOO!", kwlist, &PyArray_Type, &x_tmp, &n1_tmp, &n2_tmp, &PyArray_Type, &x_mat_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (PyIs_NativeInt(n1_tmp))
    {
        n1 = PyInt64_to_Int64(n1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n2_tmp))
    {
        n2 = PyInt64_to_Int64(n2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    bind_c_unvec_2d_omp(nd_ndim(&x, 0), nd_data(&x), n1, n2, nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_data(&x_mat));
    result = Py_BuildValue("");
    free_pointer(x);
    free_pointer(x_mat);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *vec_2d_omp_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray x_mat = {.shape = NULL};
    t_ndarray x = {.shape = NULL};
    PyArrayObject *x_mat_tmp;
    PyArrayObject *x_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "x_mat",
        "x",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist, &PyArray_Type, &x_mat_tmp, &PyArray_Type, &x_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(x_mat_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x_mat = pyarray_to_ndarray(x_mat_tmp);
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    bind_c_vec_2d_omp(nd_ndim(&x_mat, 0), nd_ndim(&x_mat, 1), nd_data(&x_mat), nd_ndim(&x, 0), nd_data(&x));
    result = Py_BuildValue("");
    free_pointer(x_mat);
    free_pointer(x);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *mxm_omp_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray A_data = {.shape = NULL};
    t_ndarray A_ind = {.shape = NULL};
    t_ndarray A_ptr = {.shape = NULL};
    t_ndarray x = {.shape = NULL};
    t_ndarray y = {.shape = NULL};
    PyArrayObject *A_data_tmp;
    PyArrayObject *A_ind_tmp;
    PyArrayObject *A_ptr_tmp;
    PyArrayObject *x_tmp;
    PyArrayObject *y_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "A_data",
        "A_ind",
        "A_ptr",
        "x",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!", kwlist, &PyArray_Type, &A_data_tmp, &PyArray_Type, &A_ind_tmp, &PyArray_Type, &A_ptr_tmp, &PyArray_Type, &x_tmp, &PyArray_Type, &y_tmp))
    {
        return NULL;
    }
    if (!pyarray_check(A_data_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_data = pyarray_to_ndarray(A_data_tmp);
    }
    if (!pyarray_check(A_ind_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_ind = pyarray_to_ndarray(A_ind_tmp);
    }
    if (!pyarray_check(A_ptr_tmp, NPY_INT32, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        A_ptr = pyarray_to_ndarray(A_ptr_tmp);
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    bind_c_mxm_omp(nd_ndim(&A_data, 0), nd_data(&A_data), nd_ndim(&A_ind, 0), nd_data(&A_ind), nd_ndim(&A_ptr, 0), nd_data(&A_ptr), nd_ndim(&x, 0), nd_ndim(&x, 1), nd_data(&x), nd_ndim(&y, 0), nd_ndim(&y, 1), nd_data(&y));
    result = Py_BuildValue("");
    free_pointer(A_data);
    free_pointer(A_ind);
    free_pointer(A_ptr);
    free_pointer(x);
    free_pointer(y);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *kron_2d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
{
    t_ndarray A1_data = {.shape = NULL};
    t_ndarray A1_ind = {.shape = NULL};
    t_ndarray A1_ptr = {.shape = NULL};
    t_ndarray A2_data = {.shape = NULL};
    t_ndarray A2_ind = {.shape = NULL};
    t_ndarray A2_ptr = {.shape = NULL};
    int64_t n_rows_1;
    int64_t n_cols_1;
    int64_t n_rows_2;
    int64_t n_cols_2;
    t_ndarray x = {.shape = NULL};
    t_ndarray W1 = {.shape = NULL};
    t_ndarray W2 = {.shape = NULL};
    t_ndarray y = {.shape = NULL};
    PyArrayObject *A1_data_tmp;
    PyArrayObject *A1_ind_tmp;
    PyArrayObject *A1_ptr_tmp;
    PyArrayObject *A2_data_tmp;
    PyArrayObject *A2_ind_tmp;
    PyArrayObject *A2_ptr_tmp;
    PyObject *n_rows_1_tmp;
    PyObject *n_cols_1_tmp;
    PyObject *n_rows_2_tmp;
    PyObject *n_cols_2_tmp;
    PyArrayObject *x_tmp;
    PyArrayObject *W1_tmp;
    PyArrayObject *W2_tmp;
    PyArrayObject *y_tmp;
    PyObject *result;
    static char *kwlist[] = {
        "A1_data",
        "A1_ind",
        "A1_ptr",
        "A2_data",
        "A2_ind",
        "A2_ptr",
        "n_rows_1",
        "n_cols_1",
        "n_rows_2",
        "n_cols_2",
        "x",
        "W1",
        "W2",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!O!OOOOO!O!O!O!", kwlist, &PyArray_Type, &A1_data_tmp, &PyArray_Type, &A1_ind_tmp, &PyArray_Type, &A1_ptr_tmp, &PyArray_Type, &A2_data_tmp, &PyArray_Type, &A2_ind_tmp, &PyArray_Type, &A2_ptr_tmp, &n_rows_1_tmp, &n_cols_1_tmp, &n_rows_2_tmp, &n_cols_2_tmp, &PyArray_Type, &x_tmp, &PyArray_Type, &W1_tmp, &PyArray_Type, &W2_tmp, &PyArray_Type, &y_tmp))
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
    if (PyIs_NativeInt(n_rows_1_tmp))
    {
        n_rows_1 = PyInt64_to_Int64(n_rows_1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_cols_1_tmp))
    {
        n_cols_1 = PyInt64_to_Int64(n_cols_1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_rows_2_tmp))
    {
        n_rows_2 = PyInt64_to_Int64(n_rows_2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_cols_2_tmp))
    {
        n_cols_2 = PyInt64_to_Int64(n_cols_2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (!pyarray_check(W1_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        W1 = pyarray_to_ndarray(W1_tmp);
    }
    if (!pyarray_check(W2_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        W2 = pyarray_to_ndarray(W2_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    bind_c_kron_2d(nd_ndim(&A1_data, 0), nd_data(&A1_data), nd_ndim(&A1_ind, 0), nd_data(&A1_ind), nd_ndim(&A1_ptr, 0), nd_data(&A1_ptr), nd_ndim(&A2_data, 0), nd_data(&A2_data), nd_ndim(&A2_ind, 0), nd_data(&A2_ind), nd_ndim(&A2_ptr, 0), nd_data(&A2_ptr), n_rows_1, n_cols_1, n_rows_2, n_cols_2, nd_ndim(&x, 0), nd_data(&x), nd_ndim(&W1, 0), nd_ndim(&W1, 1), nd_data(&W1), nd_ndim(&W2, 0), nd_ndim(&W2, 1), nd_data(&W2), nd_ndim(&y, 0), nd_data(&y));
    result = Py_BuildValue("");
    free_pointer(A1_data);
    free_pointer(A1_ind);
    free_pointer(A1_ptr);
    free_pointer(A2_data);
    free_pointer(A2_ind);
    free_pointer(A2_ptr);
    free_pointer(x);
    free_pointer(W1);
    free_pointer(W2);
    free_pointer(y);
    return result;
}
/*........................................*/

/*........................................*/
PyObject *kron_3d_wrapper(PyObject *self, PyObject *args, PyObject *kwargs)
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
    int64_t n_rows_1;
    int64_t n_cols_1;
    int64_t n_rows_2;
    int64_t n_cols_2;
    int64_t n_rows_3;
    int64_t n_cols_3;
    t_ndarray x = {.shape = NULL};
    t_ndarray Z1 = {.shape = NULL};
    t_ndarray Z2 = {.shape = NULL};
    t_ndarray Z3 = {.shape = NULL};
    t_ndarray Z4 = {.shape = NULL};
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
    PyObject *n_rows_1_tmp;
    PyObject *n_cols_1_tmp;
    PyObject *n_rows_2_tmp;
    PyObject *n_cols_2_tmp;
    PyObject *n_rows_3_tmp;
    PyObject *n_cols_3_tmp;
    PyArrayObject *x_tmp;
    PyArrayObject *Z1_tmp;
    PyArrayObject *Z2_tmp;
    PyArrayObject *Z3_tmp;
    PyArrayObject *Z4_tmp;
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
        "n_rows_1",
        "n_cols_1",
        "n_rows_2",
        "n_cols_2",
        "n_rows_3",
        "n_cols_3",
        "x",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
        "y",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!O!O!O!O!OOOOOOO!O!O!O!O!O!", kwlist, &PyArray_Type, &A1_data_tmp, &PyArray_Type, &A1_ind_tmp, &PyArray_Type, &A1_ptr_tmp, &PyArray_Type, &A2_data_tmp, &PyArray_Type, &A2_ind_tmp, &PyArray_Type, &A2_ptr_tmp, &PyArray_Type, &A3_data_tmp, &PyArray_Type, &A3_ind_tmp, &PyArray_Type, &A3_ptr_tmp, &n_rows_1_tmp, &n_cols_1_tmp, &n_rows_2_tmp, &n_cols_2_tmp, &n_rows_3_tmp, &n_cols_3_tmp, &PyArray_Type, &x_tmp, &PyArray_Type, &Z1_tmp, &PyArray_Type, &Z2_tmp, &PyArray_Type, &Z3_tmp, &PyArray_Type, &Z4_tmp, &PyArray_Type, &y_tmp))
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
    if (PyIs_NativeInt(n_rows_1_tmp))
    {
        n_rows_1 = PyInt64_to_Int64(n_rows_1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_cols_1_tmp))
    {
        n_cols_1 = PyInt64_to_Int64(n_cols_1_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_rows_2_tmp))
    {
        n_rows_2 = PyInt64_to_Int64(n_rows_2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_cols_2_tmp))
    {
        n_cols_2 = PyInt64_to_Int64(n_cols_2_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_rows_3_tmp))
    {
        n_rows_3 = PyInt64_to_Int64(n_rows_3_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (PyIs_NativeInt(n_cols_3_tmp))
    {
        n_cols_3 = PyInt64_to_Int64(n_cols_3_tmp);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "\"Argument must be native int\"");
        return NULL;
    }
    if (!pyarray_check(x_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        x = pyarray_to_ndarray(x_tmp);
    }
    if (!pyarray_check(Z1_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        Z1 = pyarray_to_ndarray(Z1_tmp);
    }
    if (!pyarray_check(Z2_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        Z2 = pyarray_to_ndarray(Z2_tmp);
    }
    if (!pyarray_check(Z3_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        Z3 = pyarray_to_ndarray(Z3_tmp);
    }
    if (!pyarray_check(Z4_tmp, NPY_DOUBLE, 2, NPY_ARRAY_C_CONTIGUOUS))
    {
        return NULL;
    }
    else
    {
        Z4 = pyarray_to_ndarray(Z4_tmp);
    }
    if (!pyarray_check(y_tmp, NPY_DOUBLE, 1, NO_ORDER_CHECK))
    {
        return NULL;
    }
    else
    {
        y = pyarray_to_ndarray(y_tmp);
    }
    bind_c_kron_3d(nd_ndim(&A1_data, 0), nd_data(&A1_data), nd_ndim(&A1_ind, 0), nd_data(&A1_ind), nd_ndim(&A1_ptr, 0), nd_data(&A1_ptr), nd_ndim(&A2_data, 0), nd_data(&A2_data), nd_ndim(&A2_ind, 0), nd_data(&A2_ind), nd_ndim(&A2_ptr, 0), nd_data(&A2_ptr), nd_ndim(&A3_data, 0), nd_data(&A3_data), nd_ndim(&A3_ind, 0), nd_data(&A3_ind), nd_ndim(&A3_ptr, 0), nd_data(&A3_ptr), n_rows_1, n_cols_1, n_rows_2, n_cols_2, n_rows_3, n_cols_3, nd_ndim(&x, 0), nd_data(&x), nd_ndim(&Z1, 0), nd_ndim(&Z1, 1), nd_data(&Z1), nd_ndim(&Z2, 0), nd_ndim(&Z2, 1), nd_data(&Z2), nd_ndim(&Z3, 0), nd_ndim(&Z3, 1), nd_data(&Z3), nd_ndim(&Z4, 0), nd_ndim(&Z4, 1), nd_data(&Z4), nd_ndim(&y, 0), nd_data(&y));
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
    free_pointer(x);
    free_pointer(Z1);
    free_pointer(Z2);
    free_pointer(Z3);
    free_pointer(Z4);
    free_pointer(y);
    return result;
}
/*........................................*/

static int exec_func(PyObject* m)
{
    return 0;
}

/*........................................*/

static PyMethodDef kronecker_csr_methods[] = {
    {
        "vec_2d",
        (PyCFunction)vec_2d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a matrix to a vector form."
    },
    {
        "unvec_2d",
        (PyCFunction)unvec_2d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a vector to a matrix form."
    },
    {
        "vec_3d",
        (PyCFunction)vec_3d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a matrix to a vector form."
    },
    {
        "unvec_3d",
        (PyCFunction)unvec_3d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a vector to a matrix form."
    },
    {
        "mxm",
        (PyCFunction)mxm_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Matrix-Vector product."
    },
    {
        "unvec_2d_omp",
        (PyCFunction)unvec_2d_omp_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a vector to a matrix form."
    },
    {
        "vec_2d_omp",
        (PyCFunction)vec_2d_omp_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Convert a matrix to a vector form."
    },
    {
        "mxm_omp",
        (PyCFunction)mxm_omp_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        "Matrix-Vector product."
    },
    {
        "kron_2d",
        (PyCFunction)kron_2d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    {
        "kron_3d",
        (PyCFunction)kron_3d_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};

/*........................................*/

static PyModuleDef_Slot kronecker_csr_slots[] = {
    {Py_mod_exec, exec_func},
    {0, NULL},
};

/*........................................*/

static struct PyModuleDef kronecker_csr_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "kronecker_csr",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    0,
    kronecker_csr_methods,
    kronecker_csr_slots
};

/*........................................*/

PyMODINIT_FUNC PyInit_kronecker_csr(void)
{
    import_array();
    return PyModuleDef_Init(&kronecker_csr_module);
}
