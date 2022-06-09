module bind_c_kronecker_csr

  use kronecker_csr, only: unvec_3d
  use kronecker_csr, only: unvec_2d
  use kronecker_csr, only: kron_2d
  use kronecker_csr, only: mxm_omp
  use kronecker_csr, only: kron_3d
  use kronecker_csr, only: unvec_2d_omp
  use kronecker_csr, only: vec_2d
  use kronecker_csr, only: mxm
  use kronecker_csr, only: vec_2d_omp
  use kronecker_csr, only: vec_3d

  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
        C_INT64_T , i32 => C_INT32_T
  implicit none

  contains

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine bind_c_vec_2d(n0_x_mat, n1_x_mat, x_mat, n0_x, x) bind(c) 

    implicit none

    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    real(f64), intent(in) :: x_mat(0:n1_x_mat - 1_i64,0:n0_x_mat - 1_i64 &
          )
    integer(i64), value :: n0_x
    real(f64), intent(inout) :: x(0:n0_x - 1_i64)

    call vec_2d(x_mat, x)

  end subroutine bind_c_vec_2d
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine bind_c_unvec_2d(n0_x, x, n1, n2, n0_x_mat, n1_x_mat, x_mat &
        ) bind(c)

    implicit none

    integer(i64), value :: n0_x
    real(f64), intent(in) :: x(0:n0_x - 1_i64)
    integer(i64), value :: n1
    integer(i64), value :: n2
    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    real(f64), intent(inout) :: x_mat(0:n1_x_mat - 1_i64,0:n0_x_mat - &
          1_i64)

    call unvec_2d(x, n1, n2, x_mat)

  end subroutine bind_c_unvec_2d
  !........................................

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine bind_c_vec_3d(n0_x_mat, n1_x_mat, n2_x_mat, x_mat, n0_x, x &
        ) bind(c)

    implicit none

    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    integer(i64), value :: n2_x_mat
    real(f64), intent(in) :: x_mat(0:n2_x_mat - 1_i64,0:n1_x_mat - 1_i64 &
          ,0:n0_x_mat - 1_i64)
    integer(i64), value :: n0_x
    real(f64), intent(inout) :: x(0:n0_x - 1_i64)

    call vec_3d(x_mat, x)

  end subroutine bind_c_vec_3d
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine bind_c_unvec_3d(n0_x, x, n1, n2, n3, n0_x_mat, n1_x_mat, &
        n2_x_mat, x_mat) bind(c)

    implicit none

    integer(i64), value :: n0_x
    real(f64), intent(in) :: x(0:n0_x - 1_i64)
    integer(i64), value :: n1
    integer(i64), value :: n2
    integer(i64), value :: n3
    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    integer(i64), value :: n2_x_mat
    real(f64), intent(inout) :: x_mat(0:n2_x_mat - 1_i64,0:n1_x_mat - &
          1_i64,0:n0_x_mat - 1_i64)

    call unvec_3d(x, n1, n2, n3, x_mat)

  end subroutine bind_c_unvec_3d
  !........................................

  !........................................
  !______________________!
  !Matrix-Vector product.!
  !______________________!

  subroutine bind_c_mxm(n0_A_data, A_data, n0_A_ind, A_ind, n0_A_ptr, &
        A_ptr, n0_x, n1_x, x, n0_y, n1_y, y) bind(c)

    implicit none

    integer(i64), value :: n0_A_data
    real(f64), intent(in) :: A_data(0:n0_A_data - 1_i64)
    integer(i64), value :: n0_A_ind
    integer(i32), intent(in) :: A_ind(0:n0_A_ind - 1_i64)
    integer(i64), value :: n0_A_ptr
    integer(i32), intent(in) :: A_ptr(0:n0_A_ptr - 1_i64)
    integer(i64), value :: n0_x
    integer(i64), value :: n1_x
    real(f64), intent(in) :: x(0:n1_x - 1_i64,0:n0_x - 1_i64)
    integer(i64), value :: n0_y
    integer(i64), value :: n1_y
    real(f64), intent(inout) :: y(0:n1_y - 1_i64,0:n0_y - 1_i64)

    call mxm(A_data, A_ind, A_ptr, x, y)

  end subroutine bind_c_mxm
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine bind_c_unvec_2d_omp(n0_x, x, n1, n2, n0_x_mat, n1_x_mat, &
        x_mat) bind(c)

    implicit none

    integer(i64), value :: n0_x
    real(f64), intent(in) :: x(0:n0_x - 1_i64)
    integer(i64), value :: n1
    integer(i64), value :: n2
    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    real(f64), intent(inout) :: x_mat(0:n1_x_mat - 1_i64,0:n0_x_mat - &
          1_i64)

    call unvec_2d_omp(x, n1, n2, x_mat)

  end subroutine bind_c_unvec_2d_omp
  !........................................

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine bind_c_vec_2d_omp(n0_x_mat, n1_x_mat, x_mat, n0_x, x) bind( &
        c)

    implicit none

    integer(i64), value :: n0_x_mat
    integer(i64), value :: n1_x_mat
    real(f64), intent(in) :: x_mat(0:n1_x_mat - 1_i64,0:n0_x_mat - 1_i64 &
          )
    integer(i64), value :: n0_x
    real(f64), intent(inout) :: x(0:n0_x - 1_i64)

    call vec_2d_omp(x_mat, x)

  end subroutine bind_c_vec_2d_omp
  !........................................

  !........................................
  !______________________!
  !Matrix-Vector product.!
  !______________________!

  subroutine bind_c_mxm_omp(n0_A_data, A_data, n0_A_ind, A_ind, n0_A_ptr &
        , A_ptr, n0_x, n1_x, x, n0_y, n1_y, y) bind(c)

    implicit none

    integer(i64), value :: n0_A_data
    real(f64), intent(in) :: A_data(0:n0_A_data - 1_i64)
    integer(i64), value :: n0_A_ind
    integer(i32), intent(in) :: A_ind(0:n0_A_ind - 1_i64)
    integer(i64), value :: n0_A_ptr
    integer(i32), intent(in) :: A_ptr(0:n0_A_ptr - 1_i64)
    integer(i64), value :: n0_x
    integer(i64), value :: n1_x
    real(f64), intent(in) :: x(0:n1_x - 1_i64,0:n0_x - 1_i64)
    integer(i64), value :: n0_y
    integer(i64), value :: n1_y
    real(f64), intent(inout) :: y(0:n1_y - 1_i64,0:n0_y - 1_i64)

    call mxm_omp(A_data, A_ind, A_ptr, x, y)

  end subroutine bind_c_mxm_omp
  !........................................

  !........................................
  subroutine bind_c_kron_2d(n0_A1_data, A1_data, n0_A1_ind, A1_ind, &
        n0_A1_ptr, A1_ptr, n0_A2_data, A2_data, n0_A2_ind, A2_ind, &
        n0_A2_ptr, A2_ptr, n_rows_1, n_cols_1, n_rows_2, n_cols_2, n0_x &
        , x, n0_W1, n1_W1, W1, n0_W2, n1_W2, W2, n0_y, y) bind(c)

    implicit none

    integer(i64), value :: n0_A1_data
    real(f64), intent(in) :: A1_data(0:n0_A1_data - 1_i64)
    integer(i64), value :: n0_A1_ind
    integer(i32), intent(in) :: A1_ind(0:n0_A1_ind - 1_i64)
    integer(i64), value :: n0_A1_ptr
    integer(i32), intent(in) :: A1_ptr(0:n0_A1_ptr - 1_i64)
    integer(i64), value :: n0_A2_data
    real(f64), intent(in) :: A2_data(0:n0_A2_data - 1_i64)
    integer(i64), value :: n0_A2_ind
    integer(i32), intent(in) :: A2_ind(0:n0_A2_ind - 1_i64)
    integer(i64), value :: n0_A2_ptr
    integer(i32), intent(in) :: A2_ptr(0:n0_A2_ptr - 1_i64)
    integer(i64), value :: n_rows_1
    integer(i64), value :: n_cols_1
    integer(i64), value :: n_rows_2
    integer(i64), value :: n_cols_2
    integer(i64), value :: n0_x
    real(f64), intent(in) :: x(0:n0_x - 1_i64)
    integer(i64), value :: n0_W1
    integer(i64), value :: n1_W1
    real(f64), intent(inout) :: W1(0:n1_W1 - 1_i64,0:n0_W1 - 1_i64)
    integer(i64), value :: n0_W2
    integer(i64), value :: n1_W2
    real(f64), intent(inout) :: W2(0:n1_W2 - 1_i64,0:n0_W2 - 1_i64)
    integer(i64), value :: n0_y
    real(f64), intent(inout) :: y(0:n0_y - 1_i64)

    call kron_2d(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, &
          n_rows_1, n_cols_1, n_rows_2, n_cols_2, x, W1, W2, y)

  end subroutine bind_c_kron_2d
  !........................................

  !........................................
  subroutine bind_c_kron_3d(n0_A1_data, A1_data, n0_A1_ind, A1_ind, &
        n0_A1_ptr, A1_ptr, n0_A2_data, A2_data, n0_A2_ind, A2_ind, &
        n0_A2_ptr, A2_ptr, n0_A3_data, A3_data, n0_A3_ind, A3_ind, &
        n0_A3_ptr, A3_ptr, n_rows_1, n_cols_1, n_rows_2, n_cols_2, &
        n_rows_3, n_cols_3, n0_x, x, n0_Z1, n1_Z1, Z1, n0_Z2, n1_Z2, Z2 &
        , n0_Z3, n1_Z3, Z3, n0_Z4, n1_Z4, Z4, n0_y, y) bind(c)

    implicit none

    integer(i64), value :: n0_A1_data
    real(f64), intent(in) :: A1_data(0:n0_A1_data - 1_i64)
    integer(i64), value :: n0_A1_ind
    integer(i32), intent(in) :: A1_ind(0:n0_A1_ind - 1_i64)
    integer(i64), value :: n0_A1_ptr
    integer(i32), intent(in) :: A1_ptr(0:n0_A1_ptr - 1_i64)
    integer(i64), value :: n0_A2_data
    real(f64), intent(in) :: A2_data(0:n0_A2_data - 1_i64)
    integer(i64), value :: n0_A2_ind
    integer(i32), intent(in) :: A2_ind(0:n0_A2_ind - 1_i64)
    integer(i64), value :: n0_A2_ptr
    integer(i32), intent(in) :: A2_ptr(0:n0_A2_ptr - 1_i64)
    integer(i64), value :: n0_A3_data
    real(f64), intent(in) :: A3_data(0:n0_A3_data - 1_i64)
    integer(i64), value :: n0_A3_ind
    integer(i32), intent(in) :: A3_ind(0:n0_A3_ind - 1_i64)
    integer(i64), value :: n0_A3_ptr
    integer(i32), intent(in) :: A3_ptr(0:n0_A3_ptr - 1_i64)
    integer(i64), value :: n_rows_1
    integer(i64), value :: n_cols_1
    integer(i64), value :: n_rows_2
    integer(i64), value :: n_cols_2
    integer(i64), value :: n_rows_3
    integer(i64), value :: n_cols_3
    integer(i64), value :: n0_x
    real(f64), intent(in) :: x(0:n0_x - 1_i64)
    integer(i64), value :: n0_Z1
    integer(i64), value :: n1_Z1
    real(f64), intent(inout) :: Z1(0:n1_Z1 - 1_i64,0:n0_Z1 - 1_i64)
    integer(i64), value :: n0_Z2
    integer(i64), value :: n1_Z2
    real(f64), intent(inout) :: Z2(0:n1_Z2 - 1_i64,0:n0_Z2 - 1_i64)
    integer(i64), value :: n0_Z3
    integer(i64), value :: n1_Z3
    real(f64), intent(inout) :: Z3(0:n1_Z3 - 1_i64,0:n0_Z3 - 1_i64)
    integer(i64), value :: n0_Z4
    integer(i64), value :: n1_Z4
    real(f64), intent(inout) :: Z4(0:n1_Z4 - 1_i64,0:n0_Z4 - 1_i64)
    integer(i64), value :: n0_y
    real(f64), intent(inout) :: y(0:n0_y - 1_i64)

    call kron_3d(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, &
          A3_data, A3_ind, A3_ptr, n_rows_1, n_cols_1, n_rows_2, &
          n_cols_2, n_rows_3, n_cols_3, x, Z1, Z2, Z3, Z4, y)

  end subroutine bind_c_kron_3d
  !........................................

end module bind_c_kronecker_csr
