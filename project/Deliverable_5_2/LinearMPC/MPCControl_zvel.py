import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        # State and input dimensions
        nx, nu = self.nx, self.nu
        N = self.N

        # Disturbance model
        self.Bd = np.ones((1, 1))

        # Tuning matrices [vz]
        self.Q = np.array([[100.0]])   
        self.R = np.array([[1.0]])     

        # Absolute thrust bounds (Pavg)
        self.umin_abs = 40.0
        self.umax_abs = 80.0

        # Terminal cost only (no terminal set)
        _, self.P, _ = dlqr(self.A, self.B, self.Q, self.R)

        # MPC in deviation coordinates
        x = cp.Variable((nx, N + 1))
        u = cp.Variable((nu, N))           # deviation input
        x0_p = cp.Parameter((nx,))         # deviation measurement
        r_p = cp.Parameter((nx,))          # deviation reference
        d_p = cp.Parameter((1,))           # disturbance estimate

        # Cost function
        cost = 0
        constr = [x[:, 0] == x0_p]
        for k in range(N):
            cost += cp.quad_form(x[:, k] - r_p, self.Q) + cp.quad_form(u[:, k], self.R)
            constr += [
                x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k] + self.Bd @ d_p,
                self.umin_abs - self.us <= u[:, k] + self.us,
                u[:, k] + self.us <= self.umax_abs,
            ]
        cost += cp.quad_form(x[:, N] - r_p, self.P)

        # Optimization problem
        self._prob = cp.Problem(cp.Minimize(cost), constr)
        self._x = x
        self._u = u
        self._x0_p = x0_p
        self._r_p = r_p
        self._d_p = d_p

        # Augmented observer
        Aaug = np.block([[self.A, self.Bd],
                         [np.zeros((1, 1)), np.ones((1, 1))]])
        Baug = np.vstack([self.B, np.zeros((1, 1))])
        Caug = np.array([[1.0, 0.0]])
        p_x = 0.30 
        p_d = 0.50 

        a11, a12 = float(Aaug[0, 0]), float(Aaug[0, 1])
        a22 = float(Aaug[1, 1]) 
        l1 = a11 + a22 - (p_x + p_d)
        l2 = (p_x * p_d - (a11 - l1) * a22) / a12
        self.L = np.array([[l1], [l2]])

        # Observer memory 
        self.x_hat = np.zeros((1, 1))
        self.d_hat = np.zeros((1, 1))
        self.u_prev = np.zeros((1, 1))  

        # For deliverable
        self.d_hat_hist = []
        self.innov_hist = [] 
        self.d_est_history = []  
        self.obs_initialized = True

    def _steady_state_input(self, r_dev: float, d_hat: float) -> tuple[float, float]:
        A = float(self.A[0, 0])
        B = float(self.B[0, 0])
        Bd = float(self.Bd[0, 0])
        u_dev = (r_dev - A * r_dev - Bd * d_hat) / B
        u_abs = u_dev + float(self.us[0])
        u_abs = float(np.clip(u_abs, self.umin_abs, self.umax_abs))

        return u_abs, float(u_dev)

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        # Measurement in deviation coordinates
        if x0.shape[0] > self.nx:
            y = np.array([[float(x0[self.x_ids][0] - self.xs[0])]])
        else:
            y = np.array([[float(x0[0] - self.xs[0])]])

        # Reference (deviation)
        if x_target is None:
            r_dev = 0.0
        else:
            if x_target.shape[0] > self.nx:
                r_dev = float(x_target[self.x_ids][0] - self.xs[0])
            else:
                r_dev = float(x_target[0] - self.xs[0])

        # Observer update
        Aaug = np.block([[self.A, self.Bd],
                         [np.zeros((1, 1)), np.ones((1, 1))]])
        Baug = np.vstack([self.B, np.zeros((1, 1))])
        Caug = np.array([[1.0, 0.0]])

        z = np.vstack([self.x_hat, self.d_hat])
        y_hat = Caug @ z
        innov = y - y_hat

        z = Aaug @ z + Baug @ self.u_prev + self.L @ innov

        self.x_hat = z[0:1, :]
        self.d_hat = z[1:2, :]

        self.d_hat_hist.append(float(self.d_hat))
        self.d_est_history.append(float(self.d_hat))
        self.innov_hist.append(float(innov))

        # Steady-state input for offset-free tracking
        u_ss_abs, u_ss_dev = self._steady_state_input(r_dev=r_dev, d_hat=float(self.d_hat))

        # Solve
        self._x0_p.value = self.x_hat.reshape(-1)
        self._r_p.value = np.array([r_dev])
        self._d_p.value = np.array([float(self.d_hat)])
        self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self._prob.status not in ("optimal", "optimal_inaccurate") or self._u.value is None:
            u_abs = u_ss_abs
            u_dev = u_ss_dev
            x_traj = np.tile(self.x_hat + self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u_abs, (1, self.N))
        else:
            u_dev = float(self._u.value[0, 0])
            u_abs = float(np.clip(u_dev + self.us[0], self.umin_abs, self.umax_abs))
            x_traj = self._x.value + self.xs.reshape(-1, 1)
            u_traj = self._u.value + self.us.reshape(-1, 1)

        # Apply and store for next observer step
        self.u_prev = np.array([[u_abs - self.us[0]]])

        return np.array([u_abs]), x_traj, u_traj
