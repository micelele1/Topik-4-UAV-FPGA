#include "ekf_types_v2.h"

/* Helper internal: baca elemen U(r, c) dengan diagonal implisit = 1.0 */
static inline ekf_u_t get_U_elem(const ekf_u_t U[U_UPPER_SIZE], int r, int c) {
    if (r == c) {
        return (ekf_u_t)1.0f;
    }
    if (r > c) {
        return (ekf_u_t)0.0f;
    }
    return U[U_IDX(r, c)];
}

/* Helper internal: proteksi underflow diagonal D */
static inline ekf_d_t clamp_D(ekf_acc_t val) {
    if (val < (ekf_acc_t)EKF_D_MIN) {
        return (ekf_d_t)EKF_D_MIN;
    }
    return (ekf_d_t)val;
}

/* ============================================================================
 * 1. INISIALISASI KOVARIANS (U = I, D = d_init, dx = 0)
 * ============================================================================ */
void ekf_algebra_init_cov(es_ekf_t *ekf, const ekf_d_t d_init[ERR_DIM]) {
    int i;
    for (i = 0; i < U_UPPER_SIZE; ++i) {
        ekf->U[i] = (ekf_u_t)0.0f;
    }
    for (i = 0; i < ERR_DIM; ++i) {
        ekf->D[i] = clamp_D((ekf_acc_t)d_init[i]);
        ekf->dx[i] = (ekf_err_t)0.0f;
    }
}

/* ============================================================================
 * 2. THORNTON MODIFIED WEIGHTED GRAM-SCHMIDT (MWGS) TIME UPDATE
 *    Menerima input dinamika terkompresi (ekf_imu_dyn_t) dari Role 1 dan
 *    membentuk blok aktif Phi secara on-the-fly tanpa matriks 15x15 penuh.
 * ============================================================================ */
void thornton_mwgs_time_update(
    es_ekf_t *ekf,
    const ekf_imu_dyn_t *dyn,
    const ekf_d_t q_d[NOISE_DIM],
    ekf_mwgs_ws_t *ws
) {
    int r, c, i, j, k, m;
    ekf_u_t *W = ws->W;
    ekf_d_t *D_bar = ws->D_bar;
    const ekf_err_t dt = dyn->dt;

    /* Blok aktif 3x3 matriks transisi Phi yang dihitung langsung dari dyn */
    ekf_err_t F_v_th[9];
    ekf_err_t F_v_ba[9];
    ekf_err_t F_th_th[9];

    const ekf_acc_t ax = (ekf_acc_t)dyn->a_unbiased[0];
    const ekf_acc_t ay = (ekf_acc_t)dyn->a_unbiased[1];
    const ekf_acc_t az = (ekf_acc_t)dyn->a_unbiased[2];

    const ekf_acc_t wx = (ekf_acc_t)dyn->w_unbiased[0];
    const ekf_acc_t wy = (ekf_acc_t)dyn->w_unbiased[1];
    const ekf_acc_t wz = (ekf_acc_t)dyn->w_unbiased[2];

    /* Bentuk blok F_v_ba = -R_q * dt dan F_v_th = -R_q * [a_unbiased]x * dt */
    for (r = 0; r < 3; ++r) {
        ekf_acc_t r0 = (ekf_acc_t)dyn->R_q[MAT3_IDX(r, 0)];
        ekf_acc_t r1 = (ekf_acc_t)dyn->R_q[MAT3_IDX(r, 1)];
        ekf_acc_t r2 = (ekf_acc_t)dyn->R_q[MAT3_IDX(r, 2)];

        F_v_ba[MAT3_IDX(r, 0)] = (ekf_err_t)(-r0 * (ekf_acc_t)dt);
        F_v_ba[MAT3_IDX(r, 1)] = (ekf_err_t)(-r1 * (ekf_acc_t)dt);
        F_v_ba[MAT3_IDX(r, 2)] = (ekf_err_t)(-r2 * (ekf_acc_t)dt);

        /* [a]x = [ 0, -az,  ay;
         *         az,   0, -ax;
         *        -ay,  ax,   0] */
        F_v_th[MAT3_IDX(r, 0)] = (ekf_err_t)(-(r1 * az - r2 * ay) * (ekf_acc_t)dt);
        F_v_th[MAT3_IDX(r, 1)] = (ekf_err_t)(-(-r0 * az + r2 * ax) * (ekf_acc_t)dt);
        F_v_th[MAT3_IDX(r, 2)] = (ekf_err_t)(-(r0 * ay - r1 * ax) * (ekf_acc_t)dt);
    }

    /* Bentuk blok F_th_th = I_3 - [w_unbiased]x * dt */
    F_th_th[0] = (ekf_err_t)1.0f;
    F_th_th[1] = (ekf_err_t)( wz * (ekf_acc_t)dt);
    F_th_th[2] = (ekf_err_t)(-wy * (ekf_acc_t)dt);

    F_th_th[3] = (ekf_err_t)(-wz * (ekf_acc_t)dt);
    F_th_th[4] = (ekf_err_t)1.0f;
    F_th_th[5] = (ekf_err_t)( wx * (ekf_acc_t)dt);

    F_th_th[6] = (ekf_err_t)( wy * (ekf_acc_t)dt);
    F_th_th[7] = (ekf_err_t)(-wx * (ekf_acc_t)dt);
    F_th_th[8] = (ekf_err_t)1.0f;

    /* Langkah 1: Bersihkan seluruh workspace W */
    for (i = 0; i < ERR_DIM * MWGS_COLS; ++i) {
        W[i] = (ekf_u_t)0.0f;
    }

    /* Langkah 2: Bentuk blok kiri W = Phi * U dengan eksploitasi sparsity */
    for (c = 0; c < ERR_DIM; ++c) {
        /* Baris 0..2 (dp): U[r, c] + dt * U[r + 3, c] */
        for (r = 0; r < 3; ++r) {
            ekf_acc_t acc = (ekf_acc_t)get_U_elem(ekf->U, r, c)
                          + (ekf_acc_t)dt * (ekf_acc_t)get_U_elem(ekf->U, r + 3, c);
            W[W_IDX(r, c)] = (ekf_u_t)acc;
        }

        /* Baris 3..5 (dv): U[r+3, c] + F_v_th * U[6..8, c] + F_v_ba * U[9..11, c] */
        for (r = 0; r < 3; ++r) {
            ekf_acc_t acc = (ekf_acc_t)get_U_elem(ekf->U, r + 3, c);
            for (m = 0; m < 3; ++m) {
                acc += (ekf_acc_t)F_v_th[MAT3_IDX(r, m)] * (ekf_acc_t)get_U_elem(ekf->U, 6 + m, c);
                acc += (ekf_acc_t)F_v_ba[MAT3_IDX(r, m)] * (ekf_acc_t)get_U_elem(ekf->U, 9 + m, c);
            }
            W[W_IDX(r + 3, c)] = (ekf_u_t)acc;
        }

        /* Baris 6..8 (dth): F_th_th * U[6..8, c] - dt * U[12..14, c] */
        for (r = 0; r < 3; ++r) {
            ekf_acc_t acc = -(ekf_acc_t)dt * (ekf_acc_t)get_U_elem(ekf->U, 12 + r, c);
            for (m = 0; m < 3; ++m) {
                acc += (ekf_acc_t)F_th_th[MAT3_IDX(r, m)] * (ekf_acc_t)get_U_elem(ekf->U, 6 + m, c);
            }
            W[W_IDX(r + 6, c)] = (ekf_u_t)acc;
        }

        /* Baris 9..14 (dba, dbg): Blok identitas, salin langsung dari U */
        for (r = 9; r < ERR_DIM; ++r) {
            W[W_IDX(r, c)] = get_U_elem(ekf->U, r, c);
        }
    }

    /* Langkah 3: Bentuk blok kanan W = G
     * - Kolom noise 0..2 (accel noise) dipetakan ke dv (baris 3..5) via R_q
     * - Kolom noise 3..5 (gyro noise) dipetakan ke dth (baris 6..8) via I_3
     * - Jika NOISE_DIM == 12:
     *   Kolom noise 6..8 (accel bias RW) -> dba (baris 9..11) via I_3
     *   Kolom noise 9..11 (gyro bias RW) -> dbg (baris 12..14) via I_3 */
    for (r = 0; r < 3; ++r) {
        for (m = 0; m < 3; ++m) {
            W[W_IDX(IDX_DV_X + r, ERR_DIM + m)] = (ekf_u_t)dyn->R_q[MAT3_IDX(r, m)];
        }
        W[W_IDX(IDX_DTH_X + r, ERR_DIM + 3 + r)] = (ekf_u_t)1.0f;
#if (NOISE_DIM == 12)
        W[W_IDX(IDX_DBA_X + r, ERR_DIM + 6 + r)] = (ekf_u_t)1.0f;
        W[W_IDX(IDX_DBG_X + r, ERR_DIM + 9 + r)] = (ekf_u_t)1.0f;
#endif
    }

    /* Langkah 4: Bentuk vektor bobot diagonal D_bar = [D | Q_d] */
    for (i = 0; i < ERR_DIM; ++i) {
        D_bar[i] = ekf->D[i];
    }
    for (i = 0; i < NOISE_DIM; ++i) {
        D_bar[ERR_DIM + i] = q_d[i];
    }

    /* Langkah 5: Ortogonalisasi MWGS (In-Place, j dari 14 turun ke 0) */
    for (j = ERR_DIM - 1; j >= 0; --j) {
        ekf_acc_t d_j_acc = (ekf_acc_t)0.0f;

        /* Hitung elemen diagonal baru: d_j = sum_k (D_bar[k] * W[j, k]^2) */
        for (k = 0; k < MWGS_COLS; ++k) {
            ekf_acc_t w_jk = (ekf_acc_t)W[W_IDX(j, k)];
            d_j_acc += (ekf_acc_t)D_bar[k] * w_jk * w_jk;
        }

        ekf_d_t d_j = clamp_D(d_j_acc);
        ekf->D[j] = d_j;

        /* Reciprocal trick: 1 kali pembagian per kolom j (15x total) */
        ekf_acc_t inv_dj = (ekf_acc_t)1.0f / (ekf_acc_t)d_j;

        for (i = 0; i < j; ++i) {
            ekf_acc_t num = (ekf_acc_t)0.0f;

            /* Weighted inner-product: sum_k (D_bar[k] * W[i, k] * W[j, k]) */
            for (k = 0; k < MWGS_COLS; ++k) {
                num += (ekf_acc_t)D_bar[k]
                     * (ekf_acc_t)W[W_IDX(i, k)]
                     * (ekf_acc_t)W[W_IDX(j, k)];
            }

            ekf_u_t u_ij = (ekf_u_t)(num * inv_dj);
            ekf->U[U_IDX(i, j)] = u_ij;

            /* Pembaruan baris W secara in-place */
            for (k = 0; k < MWGS_COLS; ++k) {
                ekf_acc_t updated = (ekf_acc_t)W[W_IDX(i, k)]
                                  - (ekf_acc_t)u_ij * (ekf_acc_t)W[W_IDX(j, k)];
                W[W_IDX(i, k)] = (ekf_u_t)updated;
            }
        }
    }
}

/* ============================================================================
 * 3. BIERMAN SCALAR OBSERVATIONAL UPDATE (VERSI UMUM / DENSE - e.g. MAG)
 *    Mengembalikan true jika diterima, false jika ditolak Outlier Gating.
 * ============================================================================ */
bool bierman_scalar_update(
    es_ekf_t *ekf,
    const ekf_err_t h[ERR_DIM],
    ekf_err_t y_tilde,
    ekf_d_t R_var,
    ekf_d_t gate_thresh
) {
    ekf_err_t f[ERR_DIM];
    ekf_acc_t g[ERR_DIM];
    ekf_acc_t k_vec[ERR_DIM];
    int i, j;

    /* Langkah 1: Hitung f = U^T * h^T, g = D * f, dan inovasi efektif */
    ekf_acc_t h_dot_dx = (ekf_acc_t)0.0f;
    ekf_acc_t S_total  = (ekf_acc_t)R_var;

    for (j = 0; j < ERR_DIM; ++j) {
        ekf_acc_t f_acc = (ekf_acc_t)h[j]; /* Diagonal U(j, j) = 1.0 implisit */
        for (i = 0; i < j; ++i) {
            f_acc += (ekf_acc_t)ekf->U[U_IDX(i, j)] * (ekf_acc_t)h[i];
        }
        f[j] = (ekf_err_t)f_acc;
        g[j] = (ekf_acc_t)ekf->D[j] * f_acc;

        S_total  += f_acc * g[j];
        h_dot_dx += (ekf_acc_t)h[j] * (ekf_acc_t)ekf->dx[j];
    }

    ekf_acc_t innov = (ekf_acc_t)y_tilde - h_dot_dx;

    /* Outlier Gating (Chi-Square) sebelum memodifikasi U, D, dx */
    if (gate_thresh > (ekf_d_t)0.0f) {
        if ((innov * innov) > ((ekf_acc_t)gate_thresh * S_total)) {
            return false;
        }
    }

    /* Langkah 2 & 3: Update faktor U, D dan vektor Kalman Gain tak ternormalisasi k */
    ekf_acc_t alpha = (R_var < (ekf_d_t)EKF_D_MIN) ? (ekf_acc_t)EKF_D_MIN : (ekf_acc_t)R_var;

    for (j = 0; j < ERR_DIM; ++j) {
        ekf_acc_t alpha_next = alpha + (ekf_acc_t)f[j] * g[j];
        if (alpha_next < (ekf_acc_t)EKF_D_MIN) {
            alpha_next = (ekf_acc_t)EKF_D_MIN;
        }

        ekf_acc_t inv_alpha = (ekf_acc_t)1.0f / alpha;
        ekf_acc_t lambda    = -(ekf_acc_t)f[j] * inv_alpha;

        ekf->D[j] = clamp_D((ekf_acc_t)ekf->D[j] * (alpha / alpha_next));

        for (i = 0; i < j; ++i) {
            ekf_u_t u_old = ekf->U[U_IDX(i, j)];
            ekf->U[U_IDX(i, j)] = (ekf_u_t)((ekf_acc_t)u_old + lambda * k_vec[i]);
            k_vec[i] += (ekf_acc_t)u_old * g[j];
        }
        k_vec[j] = g[j];
        alpha = alpha_next;
    }

    /* Langkah 4: Update Error State dx */
    ekf_acc_t scale = innov / alpha;
    for (i = 0; i < ERR_DIM; ++i) {
        ekf->dx[i] = (ekf_err_t)((ekf_acc_t)ekf->dx[i] + k_vec[i] * scale);
    }

    return true;
}

/* ============================================================================
 * 4. BIERMAN SCALAR UPDATE SPARSE (KHUSUS GNSS POS/VEL & BAROMETER)
 *    Untuk vektor Jacobian h yang hanya bernilai 1.0 pada satu indeks state_idx.
 * ============================================================================ */
bool bierman_scalar_update_sparse(
    es_ekf_t *ekf,
    uint8_t state_idx,
    ekf_err_t y_tilde,
    ekf_d_t R_var,
    ekf_d_t gate_thresh
) {
    ekf_err_t f[ERR_DIM];
    ekf_acc_t g[ERR_DIM];
    ekf_acc_t k_vec[ERR_DIM];
    int i, j;

    if (state_idx >= ERR_DIM) {
        return false;
    }

    /* Langkah 1: Eksploitasi h[state_idx] = 1.0 tanpa perkalian dot-product */
    ekf_acc_t S_total = (ekf_acc_t)R_var;

    for (j = 0; j < (int)state_idx; ++j) {
        f[j] = (ekf_err_t)0.0f;
        g[j] = (ekf_acc_t)0.0f;
        k_vec[j] = (ekf_acc_t)0.0f;
    }

    f[state_idx] = (ekf_err_t)1.0f;
    g[state_idx] = (ekf_acc_t)ekf->D[state_idx];
    S_total += g[state_idx];

    for (j = (int)state_idx + 1; j < ERR_DIM; ++j) {
        ekf_acc_t f_val = (ekf_acc_t)ekf->U[U_IDX(state_idx, j)];
        f[j] = (ekf_err_t)f_val;
        g[j] = (ekf_acc_t)ekf->D[j] * f_val;
        S_total += f_val * g[j];
    }

    ekf_acc_t innov = (ekf_acc_t)y_tilde - (ekf_acc_t)ekf->dx[state_idx];

    /* Outlier Gating */
    if (gate_thresh > (ekf_d_t)0.0f) {
        if ((innov * innov) > ((ekf_acc_t)gate_thresh * S_total)) {
            return false;
        }
    }

    /* Langkah 2 & 3: Loop dimulai langsung dari j = state_idx */
    ekf_acc_t alpha = (R_var < (ekf_d_t)EKF_D_MIN) ? (ekf_acc_t)EKF_D_MIN : (ekf_acc_t)R_var;

    for (j = (int)state_idx; j < ERR_DIM; ++j) {
        ekf_acc_t alpha_next = alpha + (ekf_acc_t)f[j] * g[j];
        if (alpha_next < (ekf_acc_t)EKF_D_MIN) {
            alpha_next = (ekf_acc_t)EKF_D_MIN;
        }

        ekf_acc_t inv_alpha = (ekf_acc_t)1.0f / alpha;
        ekf_acc_t lambda    = -(ekf_acc_t)f[j] * inv_alpha;

        ekf->D[j] = clamp_D((ekf_acc_t)ekf->D[j] * (alpha / alpha_next));

        for (i = 0; i < j; ++i) {
            ekf_u_t u_old = ekf->U[U_IDX(i, j)];
            ekf->U[U_IDX(i, j)] = (ekf_u_t)((ekf_acc_t)u_old + lambda * k_vec[i]);
            k_vec[i] += (ekf_acc_t)u_old * g[j];
        }
        k_vec[j] = g[j];
        alpha = alpha_next;
    }

    /* Langkah 4: Update Error State dx */
    ekf_acc_t scale = innov / alpha;
    for (i = 0; i < ERR_DIM; ++i) {
        ekf->dx[i] = (ekf_err_t)((ekf_acc_t)ekf->dx[i] + k_vec[i] * scale);
    }

    return true;
}

/* ============================================================================
 * 5. HELPER VERIFIKASI PC (TIDAK DISINTESIS KE FPGA)
 *    Rekonstruksi matriks kovarians penuh P = U * D * U^T untuk E_cov & NEES.
 * ============================================================================ */
#ifdef EKF_DEBUG_PC
void ekf_reconstruct_P(const es_ekf_t *ekf, double P_out[ERR_DIM * ERR_DIM]) {
    int r, c, k;
    for (r = 0; r < ERR_DIM; ++r) {
        for (c = 0; c < ERR_DIM; ++c) {
            double sum = 0.0;
            int k_start = (r > c) ? r : c;
            for (k = k_start; k < ERR_DIM; ++k) {
                double u_rk = (double)get_U_elem(ekf->U, r, k);
                double u_ck = (double)get_U_elem(ekf->U, c, k);
                sum += u_rk * (double)ekf->D[k] * u_ck;
            }
            P_out[MAT15_IDX(r, c)] = sum;
        }
    }
}
#endif
