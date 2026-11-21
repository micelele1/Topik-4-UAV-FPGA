#ifndef EKF_TYPES_H
#define EKF_TYPES_H

#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * KESEPAKATAN TEKNIKAL ROLE 3 (REVISI v2 - PASCA KONFIRMASI TIM)
 * 1. NOISE_DIM default = 12 (prioritas akurasi: 3 accel, 3 gyro, 3 accel bias RW,
 *    3 gyro bias RW). Dapat di-override saat kompilasi dengan -DNOISE_DIM=6
 *    untuk eksperimen trade-off time-accuracy.
 * 2. Input dinamika dari Role 1 menggunakan struct terkompresi ekf_imu_dyn_t
 *    (R_q, a_unbiased, w_unbiased, dt) tanpa render matriks Phi 15x15 penuh.
 * 3. Koreksi sensor Role 2 menggunakan Sequential Scalar Update (Bierman),
 *    tersedia versi dense (Magnetometer) dan sparse (GNSS Pos/Vel, Baro).
 * 4. Outlier Gating (Chi-Square) dieksekusi di dalam Bierman sebelum U, D, dx
 *    ditimpa, menggunakan parameter gate_thresh (<= 0.0f untuk nonaktif).
 * 5. Pemisahan tipe data asimetris (ekf_u_t 18-bit, ekf_d_t 27-bit,
 *    ekf_acc_t 48-bit) dipertahankan untuk kesiapan DSP48E2.
 * 6. Underflow clamping EKF_D_MIN = 1.5e-5f diterapkan pada elemen diagonal D.
 * 7. Helper rekonstruksi P = U*D*U^T dibungkus makro #ifdef EKF_DEBUG_PC.
 * 8. Matriks U disimpan murni 105 elemen (strictly upper-triangular, r < c).
 * ============================================================================ */

/* Dimensi State & Noise */
#define NOM_DIM       16
#define ERR_DIM       15

/* Default p = 12 (Akurasi tinggi). Compile dengan -DNOISE_DIM=6 untuk uji speed */
#ifndef NOISE_DIM
#define NOISE_DIM     12
#endif

#if (NOISE_DIM != 6) && (NOISE_DIM != 12)
#error "NOISE_DIM harus bernilai 6 atau 12"
#endif

#define MWGS_COLS     (ERR_DIM + NOISE_DIM)
#define U_UPPER_SIZE  105   /* ERR_DIM * (ERR_DIM - 1) / 2 (strictly r < c) */

/* Batas bawah diagonal D untuk proteksi underflow/division-by-zero */
#define EKF_D_MIN     1.5e-5f

/* Indeks Error State (dx[15]) dan Baris/Kolom Matriks 15x15 */
#define IDX_DP_X      0
#define IDX_DP_Y      1
#define IDX_DP_Z      2
#define IDX_DV_X      3
#define IDX_DV_Y      4
#define IDX_DV_Z      5
#define IDX_DTH_X     6
#define IDX_DTH_Y     7
#define IDX_DTH_Z     8
#define IDX_DBA_X     9
#define IDX_DBA_Y     10
#define IDX_DBA_Z     11
#define IDX_DBG_X     12
#define IDX_DBG_Y     13
#define IDX_DBG_Z     14

/* Makro Pengalamatan 1D Array (BRAM-Friendly) */
#define MAT3_IDX(r, c)   ((r) * 3 + (c))
#define MAT15_IDX(r, c)  ((r) * ERR_DIM + (c))
#define W_IDX(r, c)      ((r) * MWGS_COLS + (c))

/* Strictly Upper-Triangular Column-Major untuk U[105]. Syarat wajib: r < c */
#define U_IDX(r, c)      ((((c) * ((c) - 1)) / 2) + (r))

/* Pemisahan Tipe Data untuk Dual-Precision & DSP48E2 */
typedef float ekf_scalar_t; /* Kompatibilitas umum / sensor packet */
typedef float ekf_nom_t;    /* Target FPGA: 32-bit fixed-point / float32 */
typedef float ekf_err_t;    /* Target FPGA: 18-bit fixed-point */
typedef float ekf_u_t;      /* Target FPGA: ap_fixed<18, 5> */
typedef float ekf_d_t;      /* Target FPGA: ap_fixed<27, 9> */
typedef float ekf_acc_t;    /* Target FPGA: ap_fixed<48, 16> */

/* Struktur Nominal State (16-DOF) */
typedef struct {
    ekf_nom_t p[3];   /* Posisi NED [m] */
    ekf_nom_t v[3];   /* Kecepatan NED [m/s] */
    ekf_nom_t q[4];   /* Quaternion Hamilton scalar-first [qw, qx, qy, qz] */
    ekf_nom_t ba[3];  /* Bias Accel Body Frame [m/s^2] */
    ekf_nom_t bg[3];  /* Bias Gyro Body Frame [rad/s] */
} nominal_state_t;

/* Struktur Utama ES-EKF */
typedef struct {
    nominal_state_t x_nom;
    ekf_err_t dx[ERR_DIM];       /* Error state vector (15x1) */
    ekf_u_t   U[U_UPPER_SIZE];   /* Strictly upper-triangular U (diagonal = 1.0 implisit) */
    ekf_d_t   D[ERR_DIM];        /* Elemen diagonal matriks D (15x1) */
} es_ekf_t;

/* Input Dinamika Terkompresi dari Role 1 (tanpa render matriks Phi 15x15 penuh) */
typedef struct {
    ekf_err_t dt;            /* Langkah waktu IMU [s] */
    ekf_err_t R_q[9];        /* Matriks rotasi Body->NED 3x3 (row-major) */
    ekf_err_t a_unbiased[3]; /* Percepatan tanpa bias [ax, ay, az] di Body Frame [m/s^2] */
    ekf_err_t w_unbiased[3]; /* Kecepatan sudut tanpa bias [wx, wy, wz] di Body Frame [rad/s] */
} ekf_imu_dyn_t;

/* Workspace Statis untuk Thornton MWGS (Bebas malloc) */
typedef struct {
    ekf_u_t W[ERR_DIM * MWGS_COLS]; /* Matriks augmented [Phi*U | G] */
    ekf_d_t D_bar[MWGS_COLS];       /* Vektor bobot diagonal [D | Q_d] */
} ekf_mwgs_ws_t;

/* Paket Data Sensor (Role 5) */
typedef enum {
    SENS_IMU_ONLY     = 0,
    SENS_GNSS_POS_VEL = 1,
    SENS_BARO         = 2,
    SENS_MAG          = 3
} sensor_type_e;

typedef struct {
    uint32_t      timestamp_us;
    ekf_scalar_t  dt;
    sensor_type_e type;
    ekf_scalar_t  imu_acc[3];
    ekf_scalar_t  imu_gyr[3];
    ekf_scalar_t  meas[6];
} sensor_frame_t;

/* ============================================================================
 * PROTOTYPE FUNGSI ROLE 3 (ekf_algebra_v2.c)
 * ============================================================================ */

void ekf_algebra_init_cov(es_ekf_t *ekf, const ekf_d_t d_init[ERR_DIM]);

void thornton_mwgs_time_update(
    es_ekf_t *ekf,
    const ekf_imu_dyn_t *dyn,
    const ekf_d_t q_d[NOISE_DIM],
    ekf_mwgs_ws_t *ws
);

bool bierman_scalar_update(
    es_ekf_t *ekf,
    const ekf_err_t h[ERR_DIM],
    ekf_err_t y_tilde,
    ekf_d_t R_var,
    ekf_d_t gate_thresh
);

bool bierman_scalar_update_sparse(
    es_ekf_t *ekf,
    uint8_t state_idx,
    ekf_err_t y_tilde,
    ekf_d_t R_var,
    ekf_d_t gate_thresh
);

#ifdef EKF_DEBUG_PC
void ekf_reconstruct_P(const es_ekf_t *ekf, double P_out[ERR_DIM * ERR_DIM]);
#endif

#endif /* EKF_TYPES_H */
