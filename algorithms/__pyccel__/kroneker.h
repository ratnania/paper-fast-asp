#ifndef KRONEKER_H
#define KRONEKER_H

#include <stdlib.h>
#include "ndarrays.h"
#include <stdint.h>


void spsolve_kron_csr_3_sum_lower(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y);
void spsolve_kron_csr_3_sum_upper(t_ndarray A1_data, t_ndarray A1_ind, t_ndarray A1_ptr, t_ndarray A2_data, t_ndarray A2_ind, t_ndarray A2_ptr, t_ndarray A3_data, t_ndarray A3_ind, t_ndarray A3_ptr, t_ndarray B1_data, t_ndarray B1_ind, t_ndarray B1_ptr, t_ndarray B2_data, t_ndarray B2_ind, t_ndarray B2_ptr, t_ndarray B3_data, t_ndarray B3_ind, t_ndarray B3_ptr, t_ndarray C1_data, t_ndarray C1_ind, t_ndarray C1_ptr, t_ndarray C2_data, t_ndarray C2_ind, t_ndarray C2_ptr, t_ndarray C3_data, t_ndarray C3_ind, t_ndarray C3_ptr, double alpha, double beta, double gamma, t_ndarray b, t_ndarray y);
#endif // KRONEKER_H
