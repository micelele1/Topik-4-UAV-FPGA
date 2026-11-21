import numpy as np
import matplotlib.pyplot as plt
import time

# =============================================================================
# SIMULASI PERBANDINGAN ES-EKF (THORNTON MWGS + BIERMAN): p=6 vs p=12
# Konvensi: NED (World), FRD (Body), Hamilton Quaternion Scalar-First [w,x,y,z]
# =============================================================================

G_GRAV = 9.80665
EKF_D_MIN = 1.5e-5

def skew(v):
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

def quat_to_rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)

def quat_mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ], dtype=np.float64)

def quat_norm(q):
    return q / np.linalg.norm(q)

def quat_angle_error_deg(q_true, q_est):
    q_inv = np.array([q_est[0], -q_est[1], -q_est[2], -q_est[3]])
    qe = quat_mul(q_inv, q_true)
    w = np.clip(abs(qe[0]), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))

def generate_flight_data(duration=90.0, dt=0.01, seed=42):
    """
    Lintasan UAV 3D (figure-8 + climbing/descending) dimulai dari posisi Home (0,0,0)
    dengan thermal bias drift dan GPS outage selama 15 detik (t = 50s..65s).
    """
    rng = np.random.default_rng(seed)
    N = int(duration / dt)
    t = np.arange(N) * dt

    # Lintasan posisi NED (z negatif = di atas tanah)
    w_traj = 2.0 * np.pi / 40.0
    p_true = np.zeros((N, 3))
    v_true = np.zeros((N, 3))
    a_ned_true = np.zeros((N, 3))

    p_true[:, 0] = 40.0 * (1.0 - np.cos(w_traj * t))
    p_true[:, 1] = 25.0 * np.sin(2.0 * w_traj * t)
    p_true[:, 2] = -10.0 * (1.0 - np.cos(0.5 * w_traj * t))

    v_true[:, 0] = 40.0 * w_traj * np.sin(w_traj * t)
    v_true[:, 1] = 50.0 * w_traj * np.cos(2.0 * w_traj * t)
    v_true[:, 2] = -5.0 * w_traj * np.sin(0.5 * w_traj * t)

    a_ned_true[:, 0] = 40.0 * (w_traj**2) * np.cos(w_traj * t)
    a_ned_true[:, 1] = -100.0 * (w_traj**2) * np.sin(2.0 * w_traj * t)
    a_ned_true[:, 2] = -2.5 * (w_traj**2) * np.cos(0.5 * w_traj * t)

    # Orientasi (roll, pitch, yaw halus)
    q_true = np.zeros((N, 4))
    w_body_true = np.zeros((N, 3))
    q = np.array([1.0, 0.0, 0.0, 0.0])

    for k in range(N):
        tk = t[k]
        wx = 0.15 * np.cos(0.35 * tk)
        wy = 0.10 * np.sin(0.25 * tk)
        wz = 0.20 * np.cos(0.20 * tk)
        w_body_true[k] = [wx, wy, wz]
        q_true[k] = q
        dq = np.array([1.0, 0.5*wx*dt, 0.5*wy*dt, 0.5*wz*dt])
        q = quat_norm(quat_mul(q, dq))

    # True IMU bias yang bergeser perlahan (random walk + thermal drift)
    ba_true = np.zeros((N, 3))
    bg_true = np.zeros((N, 3))
    ba = np.array([0.05, -0.04, 0.06])
    bg = np.array([0.004, -0.003, 0.005])
    sigma_ba_rw = 0.020   # m/s^2 / sqrt(s)
    sigma_bg_rw = 0.0025  # rad/s / sqrt(s)

    for k in range(N):
        ba_true[k] = ba
        bg_true[k] = bg
        ba = ba + rng.normal(0.0, sigma_ba_rw * np.sqrt(dt), 3) + np.array([0.004, -0.003, 0.002]) * dt
        bg = bg + rng.normal(0.0, sigma_bg_rw * np.sqrt(dt), 3) + np.array([0.0003, -0.00025, 0.0003]) * dt

    # Pengukuran sensor mentah
    sigma_acc = 0.08
    sigma_gyr = 0.008
    sigma_gps_pos = 0.5
    sigma_gps_vel = 0.2
    sigma_baro = 0.4
    sigma_mag = 0.05
    m_ref = np.array([0.85, 0.012, 0.526])
    m_ref /= np.linalg.norm(m_ref)

    acc_meas = np.zeros((N, 3))
    gyr_meas = np.zeros((N, 3))
    gps_pos_meas = np.zeros((N, 3))
    gps_vel_meas = np.zeros((N, 3))
    baro_meas = np.zeros(N)
    mag_meas = np.zeros((N, 3))
    gps_valid = np.zeros(N, dtype=bool)
    baro_valid = np.zeros(N, dtype=bool)
    mag_valid = np.zeros(N, dtype=bool)

    g_ned = np.array([0.0, 0.0, G_GRAV])

    for k in range(N):
        R = quat_to_rot(q_true[k])
        acc_ideal = R.T @ (a_ned_true[k] - g_ned)
        acc_meas[k] = acc_ideal + ba_true[k] + rng.normal(0.0, sigma_acc, 3)
        gyr_meas[k] = w_body_true[k] + bg_true[k] + rng.normal(0.0, sigma_gyr, 3)

        # GNSS 5 Hz, mati saat GPS outage (50s <= t <= 65s)
        if (k % 20 == 0) and not (50.0 <= t[k] <= 65.0):
            gps_valid[k] = True
            gps_pos_meas[k] = p_true[k] + rng.normal(0.0, sigma_gps_pos, 3)
            gps_vel_meas[k] = v_true[k] + rng.normal(0.0, sigma_gps_vel, 3)

        # Baro & Mag 10 Hz
        if k % 10 == 0:
            baro_valid[k] = True
            baro_meas[k] = p_true[k, 2] + rng.normal(0.0, sigma_baro)
            mag_valid[k] = True
            mag_meas[k] = (R.T @ m_ref) + rng.normal(0.0, sigma_mag, 3)

    return {
        "t": t, "dt": dt, "N": N,
        "p_true": p_true, "v_true": v_true, "q_true": q_true,
        "ba_true": ba_true, "bg_true": bg_true,
        "acc_meas": acc_meas, "gyr_meas": gyr_meas,
        "gps_pos_meas": gps_pos_meas, "gps_vel_meas": gps_vel_meas, "gps_valid": gps_valid,
        "baro_meas": baro_meas, "baro_valid": baro_valid,
        "mag_meas": mag_meas, "mag_valid": mag_valid,
        "m_ref": m_ref
    }

class ESEKF_UD:
    def __init__(self, p0, v0, noise_dim=12):
        assert noise_dim in (6, 12)
        self.noise_dim = noise_dim
        self.n = 15
        self.mwgs_cols = self.n + self.noise_dim

        # Nominal state awal
        self.p = p0.copy()
        self.v = v0.copy()
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        # Error state & UDU^T factors
        self.dx = np.zeros(15)
        self.U = np.eye(15)
        self.D = np.array([
            1.0, 1.0, 1.0,
            0.5, 0.5, 0.5,
            0.05, 0.05, 0.05,
            0.01, 0.01, 0.01,
            0.0001, 0.0001, 0.0001
        ], dtype=np.float64)

        # Workspace statis
        self.W = np.zeros((15, self.mwgs_cols))
        self.D_bar = np.zeros(self.mwgs_cols)

    def predict(self, acc_m, gyr_m, dt, q_d):
        a_u = acc_m - self.ba
        w_u = gyr_m - self.bg
        R = quat_to_rot(self.q)

        # 1. Integrasi Nominal State
        a_ned = R @ a_u + np.array([0.0, 0.0, G_GRAV])
        self.p = self.p + self.v * dt + 0.5 * a_ned * dt * dt
        self.v = self.v + a_ned * dt
        dq = np.array([1.0, 0.5*w_u[0]*dt, 0.5*w_u[1]*dt, 0.5*w_u[2]*dt])
        self.q = quat_norm(quat_mul(self.q, dq))

        # 2. Thornton MWGS Time Update (Sparse Phi * U)
        t0 = time.perf_counter()
        W = self.W
        W.fill(0.0)

        F_v_th = -R @ skew(a_u) * dt
        F_v_ba = -R * dt
        F_th_th = np.eye(3) - skew(w_u) * dt

        U = self.U
        W[0:3, 0:15] = U[0:3, :] + dt * U[3:6, :]
        W[3:6, 0:15] = U[3:6, :] + F_v_th @ U[6:9, :] + F_v_ba @ U[9:12, :]
        W[6:9, 0:15] = F_th_th @ U[6:9, :] - dt * U[12:15, :]
        W[9:15, 0:15] = U[9:15, :]

        # Blok kanan W = G
        W[3:6, 15:18] = R
        W[6:9, 18:21] = np.eye(3)
        if self.noise_dim == 12:
            W[9:12, 21:24] = np.eye(3)
            W[12:15, 24:27] = np.eye(3)

        self.D_bar[:15] = self.D
        self.D_bar[15:] = q_d[:self.noise_dim]
        D_bar = self.D_bar

        # Ortogonalisasi MWGS In-Place (j = 14..0)
        for j in range(14, -1, -1):
            wj = W[j, :]
            dj = np.dot(D_bar, wj * wj)
            if dj < EKF_D_MIN:
                dj = EKF_D_MIN
            self.D[j] = dj
            inv_dj = 1.0 / dj
            if j > 0:
                dwj = D_bar * wj
                for i in range(j):
                    u_ij = np.dot(W[i, :], dwj) * inv_dj
                    U[i, j] = u_ij
                    W[i, :] -= u_ij * wj

        return time.perf_counter() - t0

    def bierman_update(self, h, y_tilde, R_var, gate_thresh=16.0):
        f = self.U.T @ h
        g = self.D * f
        S_total = R_var + np.dot(f, g)
        innov = y_tilde - np.dot(h, self.dx)

        if gate_thresh > 0.0 and (innov * innov) > (gate_thresh * S_total):
            return False

        alpha = max(R_var, EKF_D_MIN)
        k_vec = np.zeros(15)

        for j in range(15):
            alpha_next = max(alpha + f[j] * g[j], EKF_D_MIN)
            lam = -f[j] / alpha
            self.D[j] = max(self.D[j] * (alpha / alpha_next), EKF_D_MIN)
            for i in range(j):
                u_old = self.U[i, j]
                self.U[i, j] = u_old + lam * k_vec[i]
                k_vec[i] += u_old * g[j]
            k_vec[j] = g[j]
            alpha = alpha_next

        self.dx += k_vec * (innov / alpha)
        return True

    def inject_and_reset(self):
        self.p += self.dx[0:3]
        self.v += self.dx[3:6]
        dth = self.dx[6:9]
        dq = np.array([1.0, 0.5*dth[0], 0.5*dth[1], 0.5*dth[2]])
        self.q = quat_norm(quat_mul(self.q, dq))
        self.ba += self.dx[9:12]
        self.bg += self.dx[12:15]
        self.dx.fill(0.0)

def run_simulation(data, noise_dim):
    ekf = ESEKF_UD(data["p_true"][0], data["v_true"][0], noise_dim=noise_dim)
    N = data["N"]
    dt = data["dt"]
    m_ref = data["m_ref"]

    # Process noise diskrit Q_d = Q_c * dt
    q_d = np.array([
        (0.08**2)*dt, (0.08**2)*dt, (0.08**2)*dt,
        (0.008**2)*dt, (0.008**2)*dt, (0.008**2)*dt,
        (0.025**2)*dt, (0.025**2)*dt, (0.025**2)*dt,
        (0.003**2)*dt, (0.003**2)*dt, (0.003**2)*dt
    ], dtype=np.float64)

    p_est = np.zeros((N, 3))
    ba_est = np.zeros((N, 3))
    bg_est = np.zeros((N, 3))
    att_err = np.zeros(N)
    mwgs_times = np.zeros(N)

    for k in range(N):
        mwgs_times[k] = ekf.predict(data["acc_meas"][k], data["gyr_meas"][k], dt, q_d)

        updated = False
        if data["gps_valid"][k]:
            for axis in range(3):
                h = np.zeros(15); h[axis] = 1.0
                ekf.bierman_update(h, data["gps_pos_meas"][k, axis] - ekf.p[axis], 0.25)
                h = np.zeros(15); h[3 + axis] = 1.0
                ekf.bierman_update(h, data["gps_vel_meas"][k, axis] - ekf.v[axis], 0.04)
            updated = True

        if data["baro_valid"][k]:
            h = np.zeros(15); h[2] = 1.0
            ekf.bierman_update(h, data["baro_meas"][k] - ekf.p[2], 0.16)
            updated = True

        if data["mag_valid"][k]:
            R = quat_to_rot(ekf.q)
            h_mag = R.T @ m_ref
            H_mag = skew(h_mag)
            y_mag = data["mag_meas"][k] - h_mag
            for axis in range(3):
                h = np.zeros(15); h[6:9] = H_mag[axis, :]
                ekf.bierman_update(h, y_mag[axis], 0.01)
            updated = True

        if updated:
            ekf.inject_and_reset()

        p_est[k] = ekf.p
        ba_est[k] = ekf.ba
        bg_est[k] = ekf.bg
        att_err[k] = quat_angle_error_deg(data["q_true"][k], ekf.q)

    return {
        "p_est": p_est, "ba_est": ba_est, "bg_est": bg_est,
        "att_err": att_err, "mwgs_times": mwgs_times
    }

if __name__ == "__main__":
    data = generate_flight_data(duration=90.0, dt=0.01, seed=42)
    res6 = run_simulation(data, noise_dim=6)
    res12 = run_simulation(data, noise_dim=12)

    t = data["t"]
    outage = (t >= 50.0) & (t <= 65.0)

    def calc_metrics(res):
        pos_err = np.linalg.norm(res["p_est"] - data["p_true"], axis=1)
        ba_err = np.linalg.norm(res["ba_est"] - data["ba_true"], axis=1)
        bg_err = np.linalg.norm(res["bg_est"] - data["bg_true"], axis=1)
        return {
            "pos_rmse_all": np.sqrt(np.mean(pos_err**2)),
            "pos_rmse_outage": np.sqrt(np.mean(pos_err[outage]**2)),
            "pos_max_outage": np.max(pos_err[outage]),
            "att_rmse": np.sqrt(np.mean(res["att_err"]**2)),
            "ba_rmse": np.sqrt(np.mean(ba_err**2)),
            "bg_rmse": np.sqrt(np.mean(bg_err**2)),
            "mwgs_us": np.mean(res["mwgs_times"]) * 1e6,
            "pos_err": pos_err, "ba_err": ba_err, "bg_err": bg_err
        }

    m6 = calc_metrics(res6)
    m12 = calc_metrics(res12)

    print("=== RINGKASAN PERBANDINGAN p=6 vs p=12 ===")
    print(f"Position RMSE Overall (m)    : p=6 -> {m6['pos_rmse_all']:.3f} | p=12 -> {m12['pos_rmse_all']:.3f}")
    print(f"Position RMSE GPS Outage (m) : p=6 -> {m6['pos_rmse_outage']:.3f} | p=12 -> {m12['pos_rmse_outage']:.3f}")
    print(f"Position Max GPS Outage (m)  : p=6 -> {m6['pos_max_outage']:.3f} | p=12 -> {m12['pos_max_outage']:.3f}")
    print(f"Attitude RMSE (deg)          : p=6 -> {m6['att_rmse']:.3f} | p=12 -> {m12['att_rmse']:.3f}")
    print(f"Accel Bias RMSE (m/s^2)      : p=6 -> {m6['ba_rmse']:.4f} | p=12 -> {m12['ba_rmse']:.4f}")
    print(f"Gyro Bias RMSE (rad/s)       : p=6 -> {m6['bg_rmse']:.5f} | p=12 -> {m12['bg_rmse']:.5f}")
    print(f"Python MWGS avg time (us)    : p=6 -> {m6['mwgs_us']:.1f} | p=12 -> {m12['mwgs_us']:.1f}")

    # Plot hasil perbandingan
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Perbandingan ES-EKF Thornton MWGS + Bierman: p=6 vs p=12 (GPS Outage 50s-65s)", fontsize=12)

    # 1. Lintasan 2D (North-East)
    ax = axes[0, 0]
    ax.plot(data["p_true"][:, 1], data["p_true"][:, 0], 'k-', lw=2, label="Ground Truth")
    ax.plot(res6["p_est"][:, 1], res6["p_est"][:, 0], 'r--', lw=1.5, label="p=6 (Tanpa Bias RW)")
    ax.plot(res12["p_est"][:, 1], res12["p_est"][:, 0], 'b-', lw=1.5, alpha=0.85, label="p=12 (Dengan Bias RW)")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Lintasan 2D UAV (NED)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Error Posisi 3D vs Waktu
    ax = axes[0, 1]
    ax.axvspan(50.0, 65.0, color='orange', alpha=0.2, label="GPS Outage")
    ax.plot(t, m6["pos_err"], 'r-', lw=1.5, label=f"p=6 (Outage RMSE={m6['pos_rmse_outage']:.2f} m)")
    ax.plot(t, m12["pos_err"], 'b-', lw=1.5, label=f"p=12 (Outage RMSE={m12['pos_rmse_outage']:.2f} m)")
    ax.set_xlabel("Waktu [s]"); ax.set_ylabel("Error Posisi 3D [m]")
    ax.set_title("Error Posisi 3D Selama Penerbangan")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Error Estimasi Bias Akselerometer
    ax = axes[1, 0]
    ax.axvspan(50.0, 65.0, color='orange', alpha=0.2)
    ax.plot(t, m6["ba_err"], 'r-', lw=1.5, label=f"p=6 (RMSE={m6['ba_rmse']:.3f} m/s^2)")
    ax.plot(t, m12["ba_err"], 'b-', lw=1.5, label=f"p=12 (RMSE={m12['ba_rmse']:.3f} m/s^2)")
    ax.set_xlabel("Waktu [s]"); ax.set_ylabel("Error Bias Accel [m/s^2]")
    ax.set_title("Akurasi Pelacakan Bias Akselerometer")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Error Estimasi Sikap (Attitude)
    ax = axes[1, 1]
    ax.axvspan(50.0, 65.0, color='orange', alpha=0.2)
    ax.plot(t, res6["att_err"], 'r-', lw=1.5, label=f"p=6 (RMSE={m6['att_rmse']:.2f} deg)")
    ax.plot(t, res12["att_err"], 'b-', lw=1.5, label=f"p=12 (RMSE={m12['att_rmse']:.2f} deg)")
    ax.set_xlabel("Waktu [s]"); ax.set_ylabel("Error Sikap [deg]")
    ax.set_title("Error Estimasi Sikap (Attitude)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("compare_p6_vs_p12.png", dpi=150)
    print("Plot disimpan ke compare_p6_vs_p12.png")
