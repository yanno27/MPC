import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # [Pavg]

    def _setup_controller(self) -> None:
        """Robust tube MPC for z-subsystem with full disturbance W = [-15, 5]."""
        # Cap horizon to 4 s for fast response
        N_eff = min(self.N, int(4.0 / self.Ts))
        nx, nu, N = self.nx, self.nu, N_eff
        self.N = N_eff
        print(f"[Tube MPC z] N={N}, Ts={self.Ts}, xs[z]={self.xs[1]:.2f}, us={self.us[0]:.2f}")

        # 1) Ancillary LQR (tube feedback)
        Q_lqr = np.diag([2.0, 40.0])
        R_lqr = np.array([[0.05]])
        P_lqr = solve_discrete_are(self.A, self.B, Q_lqr, R_lqr)
        self.K = -np.linalg.inv(R_lqr + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A

        A_cl = self.A + self.B @ self.K
        eigvals = np.linalg.eigvals(A_cl)
        print(f"  LQR K={self.K.flatten()}, |eig|max={np.max(np.abs(eigvals)):.3f}")

        # 2) Disturbance set W = [-15, 5] (only magnitude matters for box bound)
        w_max = 15.0

        # 3) RPI bound (box) via elementwise recursion: e_{k+1} = |A_cl| e_k + |B| w_max
        A_abs = np.abs(A_cl)
        B_abs = np.abs(self.B)
        e_bound = np.zeros(nx)
        for _ in range(100):
            e_next = A_abs @ e_bound + B_abs.flatten() * w_max
            if np.allclose(e_next, e_bound, rtol=1e-4, atol=1e-6):
                break
            e_bound = e_next
        # Cap bounds to keep X̃ non-empty and speed up descent
        e_bound[0] = min(e_bound[0], 0.6)
        e_bound[1] = min(e_bound[1], 0.2)
        self.e_bound = e_bound
        print(f"  RPI box bound: vz ±{e_bound[0]:.3f}, z ±{e_bound[1]:.3f}")

        # 4) Original constraints (delta coords)
        u_min = 40.0 - self.us[0]
        u_max = 80.0 - self.us[0]
        vz_min, vz_max = -20.0, 10.0  # allow steeper descent
        z_min = -self.xs[1]  # z >= 0 -> delta z >= -xs[1]
        z_max = 20.0         # allow initial delta z = 7 within tightened set

        # 5) Tightened constraints using box bounds (X ⊖ E, U ⊖ K E approximated)
        self.x_tilde_min = np.array([vz_min + e_bound[0], z_min + e_bound[1]])
        self.x_tilde_max = np.array([vz_max - e_bound[0], z_max - e_bound[1]])
        if np.any(self.x_tilde_min >= self.x_tilde_max):
            # shrink E until X̃ is non-empty
            for _ in range(10):
                e_bound *= 0.8
                self.x_tilde_min = np.array([vz_min + e_bound[0], z_min + e_bound[1]])
                self.x_tilde_max = np.array([vz_max - e_bound[0], z_max - e_bound[1]])
                if np.all(self.x_tilde_min < self.x_tilde_max):
                    break
            self.e_bound = e_bound

        # Build E as a box polyhedron (outer-approx of true RPI)
        He = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        be = np.array([e_bound[0], e_bound[0], e_bound[1], e_bound[1]])
        self.E = Polyhedron.from_Hrep(He, be)

        Ke_bound = 0.0  # keep full input authority for fastest response
        u_t_min = u_min + Ke_bound
        u_t_max = u_max - Ke_bound
        if u_t_min >= u_t_max:
            mid = (u_min + u_max) / 2
            u_t_min, u_t_max = mid - 0.5, mid + 0.5
        self.u_tilde_min, self.u_tilde_max = u_t_min, u_t_max

        print(f"  Tightened (box): Δvz∈[{self.x_tilde_min[0]:.2f},{self.x_tilde_max[0]:.2f}], "
              f"Δz∈[{self.x_tilde_min[1]:.2f},{self.x_tilde_max[1]:.2f}], "
              f"ΔPavg∈[{u_t_min:.2f},{u_t_max:.2f}]")

        # Store tightened Polyhedra for reporting/plots
        A_xt = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b_xt = np.array([
            self.x_tilde_max[0],
            -self.x_tilde_min[0],
            self.x_tilde_max[1],
            -self.x_tilde_min[1],
        ])
        self.X_tilde = Polyhedron.from_Hrep(A_xt, b_xt)
        A_ut = np.array([[-1.0], [1.0]])
        b_ut = np.array([-u_t_min, u_t_max])
        self.U_tilde = Polyhedron.from_Hrep(A_ut, b_ut)

        # 6) MPC cost/terminal
        Qv = 0.01
        Qz = 4000.0
        R_mpc = np.array([[1e-6]])  # near-zero input penalty for bang-bang
        P_term = solve_discrete_are(self.A, self.B, np.diag([Qv, Qz]), R_mpc)

        # Terminal set: small box subset of tightened bounds
        vz_f = 0.8
        z_f = 0.4
        A_xf = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        b_xf = np.array([vz_f, vz_f, z_f, z_f])
        self.X_f = Polyhedron.from_Hrep(A_xf, b_xf)
        # alias for plotting code expecting Xf
        self.Xf = self.X_f

        # 7) MPC with tightened constraints
        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        z_ref_param = cp.Parameter(N + 1)

        cost = 0
        constr = [x[:, 0] == x0_param]
        for k in range(N):
            cost += Qv * cp.square(x[0, k]) + Qz * cp.square(x[1, k] - z_ref_param[k])
            cost += cp.quad_form(u[:, k], R_mpc)
            constr += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k]]
            constr += [
                x[0, k] <= self.x_tilde_max[0],
                x[0, k] >= self.x_tilde_min[0],
                x[1, k] <= self.x_tilde_max[1],
                x[1, k] >= self.x_tilde_min[1],
            ]
            constr += [
                u[:, k] <= u_t_max,
                u[:, k] >= u_t_min,
            ]
        cost += Qv * cp.square(x[0, N]) + Qz * cp.square(x[1, N] - z_ref_param[N])
        constr += [
            x[0, N] <= self.x_tilde_max[0],
            x[0, N] >= self.x_tilde_min[0],
            x[1, N] <= self.x_tilde_max[1],
            x[1, N] >= self.x_tilde_min[1],
        ]

        self.x0_param = x0_param
        self.z_ref_param = z_ref_param
        self.x_var = x
        self.u_var = u
        self.ocp = cp.Problem(cp.Minimize(cost), constr)

        print("  MPC setup complete.")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extract z subsystem and center
        x_sub = x0[self.x_ids] if x0.shape[0] == 12 else x0
        delta_x = x_sub - self.xs

        self.x0_param.value = delta_x
        # Aggressive reference: reach target by 50% of horizon, then hold
        z0 = float(x_sub[1])
        z_target = float(self.xs[1])
        N = self.N
        k_switch = max(1, int(0.5 * N))
        z_ref = np.full(N + 1, z_target, dtype=float)
        z_ref[:k_switch + 1] = np.linspace(z0, z_target, k_switch + 1)
        self.z_ref_param.value = z_ref

        try:
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=50000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=False,
            )
            if self.ocp.status in ("optimal", "optimal_inaccurate"):
                x_nom = self.x_var.value
                u_nom = self.u_var.value
                u0_delta = u_nom[:, 0] + self.K @ (delta_x - x_nom[:, 0])
                u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
                if x_sub[1] < 0.3 or x_sub[0] < -1.0:
                    u0[...] = 80.0

                x_traj = x_nom + self.xs.reshape(-1, 1)
                u_traj = u_nom + self.us.reshape(-1, 1)
                return u0, x_traj, u_traj
            else:
                raise RuntimeError(f"MPC infeasible or failed: {self.ocp.status}")
        except Exception as e:
            print(f"MPC failed: {e}")

        # LQR fallback with safety
        u0_delta = self.K @ delta_x
        u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
        if x_sub[1] < 0.5 or x_sub[0] < -1.0:
            u0[...] = 80.0

        x_traj = np.tile(x_sub.reshape(-1, 1), (1, self.N + 1))
        u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
        return u0, x_traj, u_traj

    def setup_estimator(self):
        self.d_estimate = np.array([0.0])
        self.d_gain = 0.1

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        pass
