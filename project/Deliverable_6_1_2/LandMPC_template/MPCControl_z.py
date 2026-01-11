import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # [Pavg]

    @staticmethod
    def _min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 40) -> Polyhedron:
        E = W
        for _ in range(max_iter):
            E_next = W + E.affine_map(A_cl)
            E_next.minHrep()
            if E_next == E:
                break
            E = E_next
        return E

    @staticmethod
    def _max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 40) -> Polyhedron:
        O = X
        for _ in range(max_iter):
            O_prev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.concatenate((f, f)))
            O.minHrep()
            if O == O_prev:
                break
        return O

    def _setup_controller(self) -> None:
        """Robust tube MPC for z-subsystem with full disturbance W = [-15, 5]."""
        nx, nu, N = self.nx, self.nu, self.N
        print(f"[Tube MPC z] N={N}, Ts={self.Ts}, xs[z]={self.xs[1]:.2f}, us={self.us[0]:.2f}")

        # 1) Ancillary LQR (tube feedback) - stronger for tighter tracking
        Q_lqr = np.diag([10.0, 80.0])
        R_lqr = np.array([[0.1]])
        P_lqr = solve_discrete_are(self.A, self.B, Q_lqr, R_lqr)
        self.K = -np.linalg.inv(R_lqr + self.B.T @ P_lqr @ self.B) @ self.B.T @ P_lqr @ self.A

        A_cl = self.A + self.B @ self.K
        eigvals = np.linalg.eigvals(A_cl)
        print(f"  LQR K={self.K.flatten()}, |eig|max={np.max(np.abs(eigvals)):.3f}")

        # 2) Disturbance set W = [-15, 5] (mapped through B)
        w_min, w_max = -15.0, 5.0
        W_w = Polyhedron.from_Hrep(A=np.array([[1.0], [-1.0]]), b=np.array([w_max, -w_min]))
        W = W_w.affine_map(self.B)

        # 3) Minimal RPI set E for x+ = A_cl x + w, w in W
        E = self._min_robust_invariant_set(A_cl, W)
        E.minHrep()
        self.E = E
        try:
            e_vz_max = E.support(np.array([1.0, 0.0]))
            e_vz_min = -E.support(np.array([-1.0, 0.0]))
            e_z_max = E.support(np.array([0.0, 1.0]))
            e_z_min = -E.support(np.array([0.0, -1.0]))
            self.e_bound = np.array([max(abs(e_vz_min), e_vz_max), max(abs(e_z_min), e_z_max)])
            print(f"  RPI bounds: vz∈[{e_vz_min:.3f},{e_vz_max:.3f}], z∈[{e_z_min:.3f},{e_z_max:.3f}]")
        except Exception:
            self.e_bound = np.array([np.nan, np.nan])
            print("  RPI bounds: (support unavailable)")

        # 4) Original constraints (delta coords)
        u_min = 40.0 - self.us[0]
        u_max = 80.0 - self.us[0]
        vz_min, vz_max = -20.0, 10.0
        z_min = -self.xs[1]  # z >= 0 -> delta z >= -xs[1]
        z_max = 20.0         # allow initial delta z = 7 within tightened set

        X = Polyhedron.from_Hrep(
            A=np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            b=np.array([vz_max, -vz_min, z_max, -z_min]),
        )
        U = Polyhedron.from_Hrep(
            A=np.array([[1.0], [-1.0]]),
            b=np.array([u_max, -u_min]),
        )

        # 5) Tightened constraints using Pontryagin differences
        X_tilde = X - E
        X_tilde.minHrep()
        KE = E.affine_map(self.K)
        U_tilde = U - KE
        U_tilde.minHrep()
        self.X_tilde = X_tilde
        self.U_tilde = U_tilde

        self.x_tilde_min = np.array([
            -X_tilde.support(np.array([-1.0, 0.0])),
            -X_tilde.support(np.array([0.0, -1.0])),
        ])
        self.x_tilde_max = np.array([
            X_tilde.support(np.array([1.0, 0.0])),
            X_tilde.support(np.array([0.0, 1.0])),
        ])
        self.u_tilde_min = -U_tilde.support(np.array([-1.0]))
        self.u_tilde_max = U_tilde.support(np.array([1.0]))

        print(f"  Tightened: Δvz∈[{self.x_tilde_min[0]:.2f},{self.x_tilde_max[0]:.2f}], "
              f"Δz∈[{self.x_tilde_min[1]:.2f},{self.x_tilde_max[1]:.2f}], "
              f"ΔPavg∈[{self.u_tilde_min:.2f},{self.u_tilde_max:.2f}]")

        # 6) Terminal set from tightened constraints
        X_tilde_and_KU_tilde = X_tilde.intersect(
            Polyhedron.from_Hrep(U_tilde.A @ self.K, U_tilde.b)
        )
        X_f = self._max_invariant_set(A_cl, X_tilde_and_KU_tilde)
        X_f.minHrep()
        self.X_f = X_f
        self.Xf = X_f

        # 7) MPC cost/terminal (aggressive position tracking)
        Q_mpc = np.diag([50.0, 3000.0])  # Very high weight on position error
        R_mpc = np.array([[0.5]])         # Moderate input penalty
        P_term = solve_discrete_are(self.A, self.B, Q_mpc, R_mpc)

        # 8) Tube MPC with tightened constraints
        z = cp.Variable((nx, N + 1))
        v = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)

        cost = 0
        constr = [self.E.A @ (x0_param - z[:, 0]) <= self.E.b]
        for k in range(N):
            cost += cp.quad_form(z[:, k], Q_mpc) + cp.quad_form(v[:, k], R_mpc)
            constr += [z[:, k + 1] == self.A @ z[:, k] + self.B @ v[:, k]]
            constr += [self.X_tilde.A @ z[:, k] <= self.X_tilde.b]
            constr += [self.U_tilde.A @ v[:, k] <= self.U_tilde.b]
        cost += cp.quad_form(z[:, N], P_term)
        constr += [self.X_f.A @ z[:, N] <= self.X_f.b]

        self.x0_param = x0_param
        self.x_var = z
        self.u_var = v
        self.ocp = cp.Problem(cp.Minimize(cost), constr)

        print("  MPC setup complete.")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extract z subsystem and center
        x_sub = x0[self.x_ids] if x0.shape[0] == 12 else x0
        delta_x = x_sub - self.xs

        self.x0_param.value = delta_x
        try:
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=40000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=False,
            )
            if self.ocp.status in ("optimal", "optimal_inaccurate"):
                z_nom = self.x_var.value
                v_nom = self.u_var.value

                # Tube MPC control law: u = v* + K(x - z*)
                u0_delta = v_nom[:, 0] + self.K @ (delta_x - z_nom[:, 0])
                u0 = np.clip(u0_delta + self.us, 40.0, 80.0)

                # Safety overrides
                z_abs = x_sub[1]
                vz_abs = x_sub[0]

                # Emergency: near ground or high velocity
                if z_abs < 0.5 or vz_abs < -3.5:
                    u0 = np.array([80.0])
                # Below target significantly - add extra thrust to compensate disturbance
                elif delta_x[1] < -0.8:  # More than 0.8m below target
                    # Add bias to counteract steady-state offset from extreme disturbance
                    u_bias = min(5.0, abs(delta_x[1]) * 3.0)  # Proportional bias
                    u0 = np.clip(u0 + u_bias, 40.0, 80.0)

                x_traj = z_nom + self.xs.reshape(-1, 1)
                u_traj = v_nom + self.us.reshape(-1, 1)
                return u0, x_traj, u_traj
            else:
                raise RuntimeError(f"MPC infeasible or failed: {self.ocp.status}")
        except Exception as e:
            print(f"MPC failed: {e}")

        # LQR fallback with aggressive safety
        z_abs = x_sub[1]
        vz_abs = x_sub[0]

        # Emergency ground safety
        if z_abs < 1.0 or vz_abs < -3.0:
            u0 = np.array([80.0])
        # Below target - climb
        elif delta_x[1] < -0.5:
            u0 = np.array([75.0])
        # Normal LQR
        else:
            u0_delta = self.K @ delta_x
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)

        x_traj = np.tile(x_sub.reshape(-1, 1), (1, self.N + 1))
        u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
        return u0, x_traj, u_traj

    def setup_estimator(self):
        self.d_estimate = np.array([0.0])
        self.d_gain = 0.1

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        pass
