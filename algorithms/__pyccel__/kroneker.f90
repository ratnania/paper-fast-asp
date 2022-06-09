module kroneker


  use, intrinsic :: ISO_C_Binding, only : i32 => C_INT32_T , i64 => &
        C_INT64_T , f64 => C_DOUBLE
  implicit none

  contains

  !........................................
  subroutine spsolve_kron_csr_3_sum_lower(A1_data, A1_ind, A1_ptr, &
        A2_data, A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, &
        B1_ind, B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, &
        B3_ptr, C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, &
        C3_data, C3_ind, C3_ptr, alpha, beta, gamma, b, y)

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
    real(f64), intent(in) :: B1_data(0:)
    integer(i32), intent(in) :: B1_ind(0:)
    integer(i32), intent(in) :: B1_ptr(0:)
    real(f64), intent(in) :: B2_data(0:)
    integer(i32), intent(in) :: B2_ind(0:)
    integer(i32), intent(in) :: B2_ptr(0:)
    real(f64), intent(in) :: B3_data(0:)
    integer(i32), intent(in) :: B3_ind(0:)
    integer(i32), intent(in) :: B3_ptr(0:)
    real(f64), intent(in) :: C1_data(0:)
    integer(i32), intent(in) :: C1_ind(0:)
    integer(i32), intent(in) :: C1_ptr(0:)
    real(f64), intent(in) :: C2_data(0:)
    integer(i32), intent(in) :: C2_ind(0:)
    integer(i32), intent(in) :: C2_ptr(0:)
    real(f64), intent(in) :: C3_data(0:)
    integer(i32), intent(in) :: C3_ind(0:)
    integer(i32), intent(in) :: C3_ptr(0:)
    real(f64), value :: alpha
    real(f64), value :: beta
    real(f64), value :: gamma
    real(f64), intent(in) :: b(0:)
    real(f64), intent(inout) :: y(0:)
    integer(i64) :: n1
    integer(i64) :: n2
    integer(i64) :: n3
    integer(i64) :: n
    integer(i64) :: i
    integer(i64) :: i1
    integer(i64) :: r
    integer(i64) :: i2
    integer(i64) :: i3
    integer(i32) :: k1_b
    integer(i32) :: k1_e
    integer(i32) :: k2_b
    integer(i32) :: k2_e
    integer(i32) :: k3_b
    integer(i32) :: k3_e
    real(f64) :: yi
    real(f64) :: a_d
    integer(i64) :: k1
    integer(i32) :: j1
    real(f64) :: a1
    integer(i64) :: k2
    integer(i32) :: j2
    real(f64) :: a2
    integer(i64) :: k3
    integer(i32) :: j3
    real(f64) :: a3
    integer(i64) :: j
    real(f64) :: zi
    real(f64) :: b_d
    real(f64) :: wi
    real(f64) :: c_d

    n1 = size(A1_ptr, kind=i64) - 1_i64
    n2 = size(A2_ptr, kind=i64) - 1_i64
    n3 = size(A3_ptr, kind=i64) - 1_i64
    n = n1 * n2 * n3
    do i = 0_i64, n - 1_i64, 1_i64
      i1 = FLOOR(i/Real((n2 * n3), f64),i64)
      r = i - i1 * n2 * n3
      i2 = FLOOR(r/Real(n3, f64),i64)
      i3 = r - i2 * n3
      !...
      k1_b = A1_ptr(i1)
      k1_e = A1_ptr(i1 + 1_i64)
      k2_b = A2_ptr(i2)
      k2_e = A2_ptr(i2 + 1_i64)
      k3_b = A3_ptr(i3)
      k3_e = A3_ptr(i3 + 1_i64)
      yi = 0.0_f64
      a_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = A1_ind(k1)
        a1 = A1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = A2_ind(k2)
          a2 = A2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = A3_ind(k3)
            a3 = A3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j < i) then
              yi = yi + a1 * a2 * a3 * y(j)
            else if (i == j) then
              a_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      !...
      k1_b = B1_ptr(i1)
      k1_e = B1_ptr(i1 + 1_i64)
      k2_b = B2_ptr(i2)
      k2_e = B2_ptr(i2 + 1_i64)
      k3_b = B3_ptr(i3)
      k3_e = B3_ptr(i3 + 1_i64)
      zi = 0.0_f64
      b_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = B1_ind(k1)
        a1 = B1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = B2_ind(k2)
          a2 = B2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = B3_ind(k3)
            a3 = B3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j < i) then
              zi = zi + a1 * a2 * a3 * y(j)
            else if (i == j) then
              b_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      !...
      k1_b = C1_ptr(i1)
      k1_e = C1_ptr(i1 + 1_i64)
      k2_b = C2_ptr(i2)
      k2_e = C2_ptr(i2 + 1_i64)
      k3_b = C3_ptr(i3)
      k3_e = C3_ptr(i3 + 1_i64)
      wi = 0.0_f64
      c_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = C1_ind(k1)
        a1 = C1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = C2_ind(k2)
          a2 = C2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = C3_ind(k3)
            a3 = C3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j < i) then
              wi = wi + a1 * a2 * a3 * y(j)
            else if (i == j) then
              c_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      y(i) = (b(i) - alpha * yi - beta * zi - gamma * wi) / (alpha * a_d &
            + beta * b_d + gamma * c_d)
    end do

  end subroutine spsolve_kron_csr_3_sum_lower
  !........................................

  !........................................
  subroutine spsolve_kron_csr_3_sum_upper(A1_data, A1_ind, A1_ptr, &
        A2_data, A2_ind, A2_ptr, A3_data, A3_ind, A3_ptr, B1_data, &
        B1_ind, B1_ptr, B2_data, B2_ind, B2_ptr, B3_data, B3_ind, &
        B3_ptr, C1_data, C1_ind, C1_ptr, C2_data, C2_ind, C2_ptr, &
        C3_data, C3_ind, C3_ptr, alpha, beta, gamma, b, y)

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
    real(f64), intent(in) :: B1_data(0:)
    integer(i32), intent(in) :: B1_ind(0:)
    integer(i32), intent(in) :: B1_ptr(0:)
    real(f64), intent(in) :: B2_data(0:)
    integer(i32), intent(in) :: B2_ind(0:)
    integer(i32), intent(in) :: B2_ptr(0:)
    real(f64), intent(in) :: B3_data(0:)
    integer(i32), intent(in) :: B3_ind(0:)
    integer(i32), intent(in) :: B3_ptr(0:)
    real(f64), intent(in) :: C1_data(0:)
    integer(i32), intent(in) :: C1_ind(0:)
    integer(i32), intent(in) :: C1_ptr(0:)
    real(f64), intent(in) :: C2_data(0:)
    integer(i32), intent(in) :: C2_ind(0:)
    integer(i32), intent(in) :: C2_ptr(0:)
    real(f64), intent(in) :: C3_data(0:)
    integer(i32), intent(in) :: C3_ind(0:)
    integer(i32), intent(in) :: C3_ptr(0:)
    real(f64), value :: alpha
    real(f64), value :: beta
    real(f64), value :: gamma
    real(f64), intent(in) :: b(0:)
    real(f64), intent(inout) :: y(0:)
    integer(i64) :: n1
    integer(i64) :: n2
    integer(i64) :: n3
    integer(i64) :: n
    integer(i64) :: i
    integer(i64) :: i1
    integer(i64) :: r
    integer(i64) :: i2
    integer(i64) :: i3
    integer(i32) :: k1_b
    integer(i32) :: k1_e
    integer(i32) :: k2_b
    integer(i32) :: k2_e
    integer(i32) :: k3_b
    integer(i32) :: k3_e
    real(f64) :: yi
    real(f64) :: a_d
    integer(i64) :: k1
    integer(i32) :: j1
    real(f64) :: a1
    integer(i64) :: k2
    integer(i32) :: j2
    real(f64) :: a2
    integer(i64) :: k3
    integer(i32) :: j3
    real(f64) :: a3
    integer(i64) :: j
    real(f64) :: zi
    real(f64) :: b_d
    real(f64) :: wi
    real(f64) :: c_d

    n1 = size(A1_ptr, kind=i64) - 1_i64
    n2 = size(A2_ptr, kind=i64) - 1_i64
    n3 = size(A3_ptr, kind=i64) - 1_i64
    n = n1 * n2 * n3
    do i = n - 1_i64, (-1_i64) + 1_i64, -1_i64
      i1 = FLOOR(i/Real((n2 * n3), f64),i64)
      r = i - i1 * n2 * n3
      i2 = FLOOR(r/Real(n3, f64),i64)
      i3 = r - i2 * n3
      !...
      k1_b = A1_ptr(i1)
      k1_e = A1_ptr(i1 + 1_i64)
      k2_b = A2_ptr(i2)
      k2_e = A2_ptr(i2 + 1_i64)
      k3_b = A3_ptr(i3)
      k3_e = A3_ptr(i3 + 1_i64)
      yi = 0.0_f64
      a_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = A1_ind(k1)
        a1 = A1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = A2_ind(k2)
          a2 = A2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = A3_ind(k3)
            a3 = A3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j >= i) then
              yi = yi + a1 * a2 * a3 * y(j)
            end if
            if (i == j) then
              a_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      !...
      k1_b = B1_ptr(i1)
      k1_e = B1_ptr(i1 + 1_i64)
      k2_b = B2_ptr(i2)
      k2_e = B2_ptr(i2 + 1_i64)
      k3_b = B3_ptr(i3)
      k3_e = B3_ptr(i3 + 1_i64)
      zi = 0.0_f64
      b_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = B1_ind(k1)
        a1 = B1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = B2_ind(k2)
          a2 = B2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = B3_ind(k3)
            a3 = B3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j >= i) then
              zi = zi + a1 * a2 * a3 * y(j)
            end if
            if (i == j) then
              b_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      !...
      k1_b = C1_ptr(i1)
      k1_e = C1_ptr(i1 + 1_i64)
      k2_b = C2_ptr(i2)
      k2_e = C2_ptr(i2 + 1_i64)
      k3_b = C3_ptr(i3)
      k3_e = C3_ptr(i3 + 1_i64)
      wi = 0.0_f64
      c_d = 1.0_f64
      do k1 = k1_b, k1_e - 1_i64, 1_i64
        j1 = C1_ind(k1)
        a1 = C1_data(k1)
        do k2 = k2_b, k2_e - 1_i64, 1_i64
          j2 = C2_ind(k2)
          a2 = C2_data(k2)
          do k3 = k3_b, k3_e - 1_i64, 1_i64
            j3 = C3_ind(k3)
            a3 = C3_data(k3)
            j = j3 + (j2 + j1 * n2) * n3
            if (j >= i) then
              wi = wi + a1 * a2 * a3 * y(j)
            end if
            if (i == j) then
              c_d = a1 * a2 * a3
            end if
          end do
        end do
      end do
      !...
      y(i) = (b(i) - alpha * yi - beta * zi - gamma * wi) / (alpha * a_d &
            + beta * b_d + gamma * c_d)
    end do

  end subroutine spsolve_kron_csr_3_sum_upper
  !........................................

end module kroneker
