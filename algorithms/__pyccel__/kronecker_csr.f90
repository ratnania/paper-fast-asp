module kronecker_csr


  use, intrinsic :: ISO_C_Binding, only : f64 => C_DOUBLE , i64 => &
        C_INT64_T , i32 => C_INT32_T
  implicit none

  contains

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine vec_2d(x_mat, x) 

    implicit none

    real(f64), intent(in) :: x_mat(0:,0:)
    real(f64), intent(inout) :: x(0:)
    integer(i64) :: n1
    integer(i64) :: n2
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i

    n1 = size(x_mat, 2_i64, i64)
    n2 = size(x_mat, 1_i64, i64)
    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        i = i2 + i1 * n2
        x(i) = x_mat(i2, i1)
      end do
    end do

  end subroutine vec_2d
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine unvec_2d(x, n1, n2, x_mat) 

    implicit none

    real(f64), intent(in) :: x(0:)
    integer(i64), value :: n1
    integer(i64), value :: n2
    real(f64), intent(inout) :: x_mat(0:,0:)
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i

    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        i = i2 + i1 * n2
        x_mat(i2, i1) = x(i)
      end do
    end do

  end subroutine unvec_2d
  !........................................

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine vec_3d(x_mat, x) 

    implicit none

    real(f64), intent(in) :: x_mat(0:,0:,0:)
    real(f64), intent(inout) :: x(0:)
    integer(i64) :: n1
    integer(i64) :: n2
    integer(i64) :: n3
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i3
    integer(i64) :: i

    n1 = size(x_mat, 3_i64, i64)
    n2 = size(x_mat, 2_i64, i64)
    n3 = size(x_mat, 1_i64, i64)
    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        do i3 = 0_i64, n3 - 1_i64, 1_i64
          i = i3 + (i2 + i1 * n2) * n3
          x(i) = x_mat(i3, i2, i1)
        end do
      end do
    end do

  end subroutine vec_3d
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine unvec_3d(x, n1, n2, n3, x_mat) 

    implicit none

    real(f64), intent(in) :: x(0:)
    integer(i64), value :: n1
    integer(i64), value :: n2
    integer(i64), value :: n3
    real(f64), intent(inout) :: x_mat(0:,0:,0:)
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i3
    integer(i64) :: i

    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        do i3 = 0_i64, n3 - 1_i64, 1_i64
          i = i3 + (i2 + i1 * n2) * n3
          x_mat(i3, i2, i1) = x(i)
        end do
      end do
    end do

  end subroutine unvec_3d
  !........................................

  !........................................
  !______________________!
  !Matrix-Vector product.!
  !______________________!

  subroutine mxm(A_data, A_ind, A_ptr, x, y) 

    implicit none

    real(f64), intent(in) :: A_data(0:)
    integer(i32), intent(in) :: A_ind(0:)
    integer(i32), intent(in) :: A_ptr(0:)
    real(f64), intent(in) :: x(0:,0:)
    real(f64), intent(inout) :: y(0:,0:)
    integer(i64) :: n
    integer(i64) :: m
    integer(i64) :: k
    integer(i64) :: i
    real(f64) :: wi
    integer(i64) :: j

    n = size(A_ptr, kind=i64) - 1_i64
    m = size(x, 2_i64, i64)
    do k = 0_i64, m - 1_i64, 1_i64
      do i = 0_i64, n - 1_i64, 1_i64
        wi = 0.0_f64
        do j = A_ptr(i), A_ptr(i + 1_i64) - 1_i64, 1_i64
          wi = wi + A_data(j) * x(A_ind(j), k)
        end do
        y(k, i) = wi
      end do
    end do

  end subroutine mxm
  !........................................

  !........................................
  !__________________________________!
  !Convert a vector to a matrix form.!
  !__________________________________!

  subroutine unvec_2d_omp(x, n1, n2, x_mat) 

    implicit none

    real(f64), intent(in) :: x(0:)
    integer(i64), value :: n1
    integer(i64), value :: n2
    real(f64), intent(inout) :: x_mat(0:,0:)
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i

    !$omp do schedule(runtime)
    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        i = i2 + i1 * n2
        x_mat(i2, i1) = x(i)
      end do
    end do
    !$omp end do

  end subroutine unvec_2d_omp
  !........................................

  !........................................
  !__________________________________!
  !Convert a matrix to a vector form.!
  !__________________________________!

  subroutine vec_2d_omp(x_mat, x) 

    implicit none

    real(f64), intent(in) :: x_mat(0:,0:)
    real(f64), intent(inout) :: x(0:)
    integer(i64) :: n1
    integer(i64) :: n2
    integer(i64) :: i1
    integer(i64) :: i2
    integer(i64) :: i

    n1 = size(x_mat, 2_i64, i64)
    n2 = size(x_mat, 1_i64, i64)
    !$omp do schedule(runtime)
    do i1 = 0_i64, n1 - 1_i64, 1_i64
      do i2 = 0_i64, n2 - 1_i64, 1_i64
        i = i2 + i1 * n2
        x(i) = x_mat(i2, i1)
      end do
    end do
    !$omp end do

  end subroutine vec_2d_omp
  !........................................

  !........................................
  !______________________!
  !Matrix-Vector product.!
  !______________________!

  subroutine mxm_omp(A_data, A_ind, A_ptr, x, y) 

    implicit none

    real(f64), intent(in) :: A_data(0:)
    integer(i32), intent(in) :: A_ind(0:)
    integer(i32), intent(in) :: A_ptr(0:)
    real(f64), intent(in) :: x(0:,0:)
    real(f64), intent(inout) :: y(0:,0:)
    integer(i64) :: n
    integer(i64) :: m
    integer(i64) :: k
    integer(i64) :: i
    real(f64) :: wi
    integer(i64) :: j

    n = size(A_ptr, kind=i64) - 1_i64
    m = size(x, 2_i64, i64)
    !$omp do schedule(runtime)
    do k = 0_i64, m - 1_i64, 1_i64
      do i = 0_i64, n - 1_i64, 1_i64
        wi = 0.0_f64
        do j = A_ptr(i), A_ptr(i + 1_i64) - 1_i64, 1_i64
          wi = wi + A_data(j) * x(A_ind(j), k)
        end do
        y(k, i) = wi
      end do
    end do
    !$omp end do

  end subroutine mxm_omp
  !........................................

  !........................................
  subroutine kron_2d(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, &
        n_rows_1, n_cols_1, n_rows_2, n_cols_2, x, W1, W2, y)

    implicit none

    real(f64), intent(in) :: A1_data(0:)
    integer(i32), intent(in) :: A1_ind(0:)
    integer(i32), intent(in) :: A1_ptr(0:)
    real(f64), intent(in) :: A2_data(0:)
    integer(i32), intent(in) :: A2_ind(0:)
    integer(i32), intent(in) :: A2_ptr(0:)
    integer(i64), value :: n_rows_1
    integer(i64), value :: n_cols_1
    integer(i64), value :: n_rows_2
    integer(i64), value :: n_cols_2
    real(f64), intent(in) :: x(0:)
    real(f64), intent(inout) :: W1(0:,0:)
    real(f64), intent(inout) :: W2(0:,0:)
    real(f64), intent(inout) :: y(0:)

    !...
    call unvec_2d(x, n_cols_1, n_cols_2, W1(0_i64:n_cols_2 - 1_i64, &
          0_i64:n_cols_1 - 1_i64))
    !...
    !print(W1)
    !...
    call mxm(A2_data, A2_ind, A2_ptr, W1(:n_cols_2 - 1_i64, :n_cols_1 - &
          1_i64), W2(:n_cols_1 - 1_i64, :n_rows_2 - 1_i64))
    !...
    !print(W2)
    !...
    call mxm(A1_data, A1_ind, A1_ptr, W2(:n_cols_1 - 1_i64, :n_rows_2 - &
          1_i64), W1(:n_rows_2 - 1_i64, :n_rows_1 - 1_i64))
    !...
    !...
    call vec_2d(W1(0_i64:n_rows_2 - 1_i64, 0_i64:n_rows_1 - 1_i64), y)
    !...

  end subroutine kron_2d
  !........................................

  !........................................
  subroutine kron_3d(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, &
        A3_data, A3_ind, A3_ptr, n_rows_1, n_cols_1, n_rows_2, n_cols_2 &
        , n_rows_3, n_cols_3, x, Z1, Z2, Z3, Z4, y)

    implicit none

    real(f64), intent(in) :: A1_data(0:)
    integer(i32), intent(in) :: A1_ind(0:)
    integer(i32), intent(in) :: A1_ptr(0:)
    real(f64), intent(in) :: A2_data(0:)
    integer(i32), intent(in) :: A2_ind(0:)
    integer(i32), intent(in) :: A2_ptr(0:)
    real(f64), intent(in) :: A3_data(0:)
    integer(i32), intent(in) :: A3_ind(0:)
    integer(i32), intent(in) :: A3_ptr(0:)
    integer(i64), value :: n_rows_1
    integer(i64), value :: n_cols_1
    integer(i64), value :: n_rows_2
    integer(i64), value :: n_cols_2
    integer(i64), value :: n_rows_3
    integer(i64), value :: n_cols_3
    real(f64), intent(in) :: x(0:)
    real(f64), intent(inout) :: Z1(0:,0:)
    real(f64), intent(inout) :: Z2(0:,0:)
    real(f64), intent(inout) :: Z3(0:,0:)
    real(f64), intent(inout) :: Z4(0:,0:)
    real(f64), intent(inout) :: y(0:)
    integer(i64) :: n_rows_12
    integer(i64) :: n_cols_12
    integer(i64) :: k

    !$omp parallel
    n_rows_12 = n_rows_1 * n_rows_2
    n_cols_12 = n_cols_1 * n_cols_2
    call unvec_2d_omp(x, n_cols_12, n_cols_3, Z1(0_i64:n_cols_3 - 1_i64, &
          0_i64:n_cols_12 - 1_i64))
    call mxm_omp(A3_data, A3_ind, A3_ptr, Z1(:n_cols_3 - 1_i64, : &
          n_cols_12 - 1_i64), Z2(:n_cols_12 - 1_i64, :n_rows_3 - 1_i64 &
          ))
    !$omp do schedule(runtime) private(k, Z3, Z4)
    do k = 0_i64, n_rows_3 - 1_i64, 1_i64
      call kron_2d(A1_data, A1_ind, A1_ptr, A2_data, A2_ind, A2_ptr, &
            n_rows_1, n_cols_1, n_rows_2, n_cols_2, Z2(0_i64:n_cols_12 &
            - 1_i64, k), Z3, Z4, Z1(k, 0_i64:n_rows_12 - 1_i64))
    end do
    !$omp end do
    call vec_2d_omp(Z1(0_i64:n_rows_3 - 1_i64, 0_i64:n_rows_12 - 1_i64), &
          y)
    !$omp end parallel

  end subroutine kron_3d
  !........................................

end module kronecker_csr
