module bind_c_kroneker

  use kroneker, only: spsolve_kron_csr_3_sum_upper
  use kroneker, only: spsolve_kron_csr_3_sum_lower

  use, intrinsic :: ISO_C_Binding, only : i32 => C_INT32_T , i64 => &
        C_INT64_T , f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine bind_c_spsolve_kron_csr_3_sum_lower(n0_A1_data, A1_data, &
        n0_A1_ind, A1_ind, n0_A1_ptr, A1_ptr, n0_A2_data, A2_data, &
        n0_A2_ind, A2_ind, n0_A2_ptr, A2_ptr, n0_A3_data, A3_data, &
        n0_A3_ind, A3_ind, n0_A3_ptr, A3_ptr, n0_B1_data, B1_data, &
        n0_B1_ind, B1_ind, n0_B1_ptr, B1_ptr, n0_B2_data, B2_data, &
        n0_B2_ind, B2_ind, n0_B2_ptr, B2_ptr, n0_B3_data, B3_data, &
        n0_B3_ind, B3_ind, n0_B3_ptr, B3_ptr, n0_C1_data, C1_data, &
        n0_C1_ind, C1_ind, n0_C1_ptr, C1_ptr, n0_C2_data, C2_data, &
        n0_C2_ind, C2_ind, n0_C2_ptr, C2_ptr, n0_C3_data, C3_data, &
        n0_C3_ind, C3_ind, n0_C3_ptr, C3_ptr, alpha, beta, gamma, n0_b, &
        b, n0_y, y) bind(c)

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
    integer(i64), value :: n0_B1_data
    real(f64), intent(in) :: B1_data(0:n0_B1_data - 1_i64)
    integer(i64), value :: n0_B1_ind
    integer(i32), intent(in) :: B1_ind(0:n0_B1_ind - 1_i64)
    integer(i64), value :: n0_B1_ptr
    integer(i32), intent(in) :: B1_ptr(0:n0_B1_ptr - 1_i64)
    integer(i64), value :: n0_B2_data
    real(f64), intent(in) :: B2_data(0:n0_B2_data - 1_i64)
    integer(i64), value :: n0_B2_ind
    integer(i32), intent(in) :: B2_ind(0:n0_B2_ind - 1_i64)
    integer(i64), value :: n0_B2_ptr
    integer(i32), intent(in) :: B2_ptr(0:n0_B2_ptr - 1_i64)
    integer(i64), value :: n0_B3_data
    real(f64), intent(in) :: B3_data(0:n0_B3_data - 1_i64)
    integer(i64), value :: n0_B3_ind
    integer(i32), intent(in) :: B3_ind(0:n0_B3_ind - 1_i64)
    integer(i64), value :: n0_B3_ptr
    integer(i32), intent(in) :: B3_ptr(0:n0_B3_ptr - 1_i64)
    integer(i64), value :: n0_C1_data
    real(f64), intent(in) :: C1_data(0:n0_C1_data - 1_i64)
    integer(i64), value :: n0_C1_ind
    integer(i32), intent(in) :: C1_ind(0:n0_C1_ind - 1_i64)
    integer(i64), value :: n0_C1_ptr
    integer(i32), intent(in) :: C1_ptr(0:n0_C1_ptr - 1_i64)
    integer(i64), value :: n0_C2_data
    real(f64), intent(in) :: C2_data(0:n0_C2_data - 1_i64)
    integer(i64), value :: n0_C2_ind
    integer(i32), intent(in) :: C2_ind(0:n0_C2_ind - 1_i64)
    integer(i64), value :: n0_C2_ptr
    integer(i32), intent(in) :: C2_ptr(0:n0_C2_ptr - 1_i64)
    integer(i64), value :: n0_C3_data
    real(f64), intent(in) :: C3_data(0:n0_C3_data - 1_i64)
    integer(i64), value :: n0_C3_ind
    integer(i32), intent(in) :: C3_ind(0:n0_C3_ind - 1_i64)
    integer(i64), value :: n0_C3_ptr
    integer(i32), intent(in) :: C3_ptr(0:n0_C3_ptr - 1_i64)
    real(f64), value :: alpha
    real(f64), value :: beta
    real(f64), value :: gamma
    integer(i64), value :: n0_b
    real(f64), intent(in) :: b(0:n0_b - 1_i64)
    integer(i64), value :: n0_y
    real(f64), intent(inout) :: y(0:n0_y - 1_i64)

    call spsolve_kron_csr_3_sum_lower(A1_data, A1_ind, A1_ptr, A2_data, &
          A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, B1_ind, &
          B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, B3_ptr, &
          C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, C3_data, &
          C3_ind, C3_ptr, alpha, beta, gamma, b, y)

  end subroutine bind_c_spsolve_kron_csr_3_sum_lower
  !........................................

  !........................................
  subroutine bind_c_spsolve_kron_csr_3_sum_upper(n0_A1_data, A1_data, &
        n0_A1_ind, A1_ind, n0_A1_ptr, A1_ptr, n0_A2_data, A2_data, &
        n0_A2_ind, A2_ind, n0_A2_ptr, A2_ptr, n0_A3_data, A3_data, &
        n0_A3_ind, A3_ind, n0_A3_ptr, A3_ptr, n0_B1_data, B1_data, &
        n0_B1_ind, B1_ind, n0_B1_ptr, B1_ptr, n0_B2_data, B2_data, &
        n0_B2_ind, B2_ind, n0_B2_ptr, B2_ptr, n0_B3_data, B3_data, &
        n0_B3_ind, B3_ind, n0_B3_ptr, B3_ptr, n0_C1_data, C1_data, &
        n0_C1_ind, C1_ind, n0_C1_ptr, C1_ptr, n0_C2_data, C2_data, &
        n0_C2_ind, C2_ind, n0_C2_ptr, C2_ptr, n0_C3_data, C3_data, &
        n0_C3_ind, C3_ind, n0_C3_ptr, C3_ptr, alpha, beta, gamma, n0_b, &
        b, n0_y, y) bind(c)

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
    integer(i64), value :: n0_B1_data
    real(f64), intent(in) :: B1_data(0:n0_B1_data - 1_i64)
    integer(i64), value :: n0_B1_ind
    integer(i32), intent(in) :: B1_ind(0:n0_B1_ind - 1_i64)
    integer(i64), value :: n0_B1_ptr
    integer(i32), intent(in) :: B1_ptr(0:n0_B1_ptr - 1_i64)
    integer(i64), value :: n0_B2_data
    real(f64), intent(in) :: B2_data(0:n0_B2_data - 1_i64)
    integer(i64), value :: n0_B2_ind
    integer(i32), intent(in) :: B2_ind(0:n0_B2_ind - 1_i64)
    integer(i64), value :: n0_B2_ptr
    integer(i32), intent(in) :: B2_ptr(0:n0_B2_ptr - 1_i64)
    integer(i64), value :: n0_B3_data
    real(f64), intent(in) :: B3_data(0:n0_B3_data - 1_i64)
    integer(i64), value :: n0_B3_ind
    integer(i32), intent(in) :: B3_ind(0:n0_B3_ind - 1_i64)
    integer(i64), value :: n0_B3_ptr
    integer(i32), intent(in) :: B3_ptr(0:n0_B3_ptr - 1_i64)
    integer(i64), value :: n0_C1_data
    real(f64), intent(in) :: C1_data(0:n0_C1_data - 1_i64)
    integer(i64), value :: n0_C1_ind
    integer(i32), intent(in) :: C1_ind(0:n0_C1_ind - 1_i64)
    integer(i64), value :: n0_C1_ptr
    integer(i32), intent(in) :: C1_ptr(0:n0_C1_ptr - 1_i64)
    integer(i64), value :: n0_C2_data
    real(f64), intent(in) :: C2_data(0:n0_C2_data - 1_i64)
    integer(i64), value :: n0_C2_ind
    integer(i32), intent(in) :: C2_ind(0:n0_C2_ind - 1_i64)
    integer(i64), value :: n0_C2_ptr
    integer(i32), intent(in) :: C2_ptr(0:n0_C2_ptr - 1_i64)
    integer(i64), value :: n0_C3_data
    real(f64), intent(in) :: C3_data(0:n0_C3_data - 1_i64)
    integer(i64), value :: n0_C3_ind
    integer(i32), intent(in) :: C3_ind(0:n0_C3_ind - 1_i64)
    integer(i64), value :: n0_C3_ptr
    integer(i32), intent(in) :: C3_ptr(0:n0_C3_ptr - 1_i64)
    real(f64), value :: alpha
    real(f64), value :: beta
    real(f64), value :: gamma
    integer(i64), value :: n0_b
    real(f64), intent(in) :: b(0:n0_b - 1_i64)
    integer(i64), value :: n0_y
    real(f64), intent(inout) :: y(0:n0_y - 1_i64)

    call spsolve_kron_csr_3_sum_upper(A1_data, A1_ind, A1_ptr, A2_data, &
          A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, B1_ind, &
          B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, B3_ptr, &
          C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, C3_data, &
          C3_ind, C3_ptr, alpha, beta, gamma, b, y)

  end subroutine bind_c_spsolve_kron_csr_3_sum_upper
  !........................................

end module bind_c_kroneker
