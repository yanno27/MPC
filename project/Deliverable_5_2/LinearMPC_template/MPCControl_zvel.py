import numpy as np
import cvxpy as cp
import numpy as np
from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        nx, nu = self.nx, self.nu
        N = self.N

        # disturbance observer state (x_hat, d_hat) and previous input
        self.x_hat = np.zeros(nx)
        self.d_hat = np.zeros(nx)
        self.u_prev = np.zeros(nu)
        self.obs_initialized = False
        self.d_est_history = []
        self.error_integral = np.zeros(nx)

        # costs
        Q = np.array([[100.0]])
        R = np.array([[1.0]])

        # input constraints in delta form (absolute bounds 40-80%)
        u_min = 40.0 - self.us[0]
        u_max = 80.0 - self.us[0]

        # MPC variables (no terminal set, stage cost only)
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        x_ref_param = cp.Parameter(nx)

        cost = 0
        constraints = [x_var[:, 0] == x0_param]
        for k in range(N):
            constraints += [x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]]
            constraints += [u_var[:, k] >= u_min, u_var[:, k] <= u_max]
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q) + cp.quad_form(u_var[:, k], R)
        cost += cp.quad_form(x_var[:, N] - x_ref_param, Q)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.x_ref_param = x_ref_param

        # simple observer gain for [x_hat; d_hat]
        self.L = np.array([[0.25],   # on x error
                           [0.01]]) # on x error into d_hat

    def get_u(self, x0, x_target=None, u_target=None):
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
        
        if x_target is not None and x_target.shape[0] > self.nx:
            x_target = x_target[self.x_ids]
        
        if x_target is None:
            x_ref = self.xs
        else:
            x_ref = x_target
        
        # ---------- Disturbance observer (constant d) ----------
        y = x0 - self.xs  # measured delta vz
        if not self.obs_initialized:
            self.x_hat = y.copy()
            self.d_hat = np.zeros_like(y)
            self.u_prev = np.zeros(self.nu)
            self.obs_initialized = True
        # augmented state [x_hat; d_hat]
        z = np.vstack([self.x_hat.reshape(-1,1), self.d_hat.reshape(-1,1)])
        A_aug = np.block([[self.A, np.eye(self.nx)],
                          [np.zeros((self.nx, self.nx)), np.eye(self.nx)]])
        B_aug = np.vstack([self.B, np.zeros((self.nx, self.nu))])
        C_aug = np.hstack([np.eye(self.nx), np.zeros((self.nx, self.nx))])
        y_hat = C_aug @ z
        err = (y.reshape(-1,1) - y_hat)
        z = A_aug @ z + B_aug @ self.u_prev.reshape(-1,1) + self.L @ err
        self.x_hat = z[:self.nx].flatten()
        self.d_hat = z[self.nx:].flatten()
        self.d_est_history.append(self.d_hat.copy())
        self.error_integral += err.flatten() * self.Ts

        # steady-state offset compensation for constant d: x_ss = d / (1 - A)
        A_scalar = float(self.A)
        off = self.d_hat / (1.0 - A_scalar) if abs(1.0 - A_scalar) > 1e-6 else 0.0

        delta_x0 = self.x_hat.copy()
        delta_x_ref = (x_ref - self.xs) - off

        self.x0_param.value = delta_x0
        self.x_ref_param.value = delta_x_ref
        
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        
        if self.ocp.status != cp.OPTIMAL:
            print(f"[Z-vel] Warning: status={self.ocp.status}")
            print(f"  delta_x0={delta_x0[0]:.3f}, d_hat={self.d_hat[0]:.3f}")
            # Return trim input (not zero!) to avoid constraint violation
            u0 = self.us
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj
        
        delta_u_opt = self.u_var.value
        delta_x_opt = self.x_var.value
        self.u_prev = delta_u_opt[:, 0].copy()

        u0_raw = delta_u_opt[:, 0] + self.us
        # Clamp to respect absolute input constraints (40-80%)
        u0 = np.clip(u0_raw, 40.0, 80.0)

        x_traj = delta_x_opt + self.xs.reshape(-1, 1)
        u_traj = delta_u_opt + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj
