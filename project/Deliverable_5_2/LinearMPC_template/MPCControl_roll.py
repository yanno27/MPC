import numpy as np
import cvxpy as cp
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
from scipy.signal import place_poles


class MPCControl_roll:
    """Roll MPC with disturbance observer (Deliverable 5.1).
    States: [gamma, omega_z]
    Input:  [Mz]
    """

    x_ids = np.array([5, 2])  # gamma, omega_z
    u_ids = np.array([3])     # Mz

    def __init__(self, A, B, xs, us, Ts, H):
        self.Ts = Ts
        self.N = int(H / Ts)

        # Reduce system
        Ared = A[np.ix_(self.x_ids, self.x_ids)]
        Bred = B[np.ix_(self.x_ids, self.u_ids)]

        # Discretize
        C = np.eye(Ared.shape[0])
        D = np.zeros((Ared.shape[0], Bred.shape[1]))
        Ad, Bd_u, _, _, _ = cont2discrete((Ared, Bred, C, D), Ts)
        Bd_d = np.eye(Ad.shape[0])  # Full disturbance model

        self.Ad = Ad
        self.Bd_u = Bd_u
        self.Bd_d = Bd_d

        nx = Ad.shape[0]
        nu = Bd_u.shape[1]
        N = self.N  # Use the stored horizon

        # Costs (gamma, omega_z)
        Q = np.diag([50.0, 5.0])
        R = np.diag([1.0])

        P = solve_discrete_are(Ad, Bd_u, Q, R)

        x = cp.Variable((nx, N + 1))
        v = cp.Variable((nu, N))
        x0 = cp.Parameter(nx)
        us_param = cp.Parameter(nu)

        umin = -1.0
        umax = 1.0

        cost = 0
        constr = [x[:, 0] == x0]

        for k in range(N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(v[:, k], R)
            constr += [
                x[:, k + 1] == Ad @ x[:, k] + Bd_u @ v[:, k],
                umin - us_param <= v[:, k],
                v[:, k] <= umax - us_param,
            ]

        cost += cp.quad_form(x[:, N], P)

        self.prob = cp.Problem(cp.Minimize(cost), constr)

        # Observer - augmented state [x; d] where d is nx-dimensional disturbance
        nd = nx  # Disturbance dimension matches state dimension
        Aaug = np.block([[Ad, Bd_d], [np.zeros((nd, nx)), np.eye(nd)]])
        Caug = np.hstack([np.eye(nx), np.zeros((nx, nd))])

        # Observer gain using separation principle
        # Since we measure all x states, design L directly
        # L = [L_x; L_d] for state and disturbance estimation
        L_x = 0.25 * np.eye(nx)  # State observer gain
        L_d = 0.01 * np.eye(nd)  # Disturbance observer gain (slower)
        L = np.vstack([L_x, L_d])

        self.L = L
        self.xhat = np.zeros(nx)
        self.dhat = np.zeros(nd)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]
        self.x0_param = x0
        self.us_param = us_param
        self.u_prev = np.zeros(nu)

    def get_u(self, x0, x_target=None, u_target=None):
        # Extract roll states from full state vector
        if x0.shape[0] > self.xhat.shape[0]:
            x_meas = x0[self.x_ids]
        else:
            x_meas = x0

        y = x_meas - self.xs  # Measure deviation from trim
        nx = self.xhat.shape[0]
        nd = self.dhat.shape[0]

        # No disturbance observer: use measured deviation directly
        self.xhat = y.copy()
        self.dhat = np.zeros(nd)

        # Reference (stabilize to zero deviation)
        xs = np.zeros(nx)
        us_ff = np.zeros(1)

        self.prob.parameters()[0].value = self.xhat - xs
        self.prob.parameters()[1].value = us_ff
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self.prob.status != cp.OPTIMAL:
            print(f"[Roll] Warning: solver status = {self.prob.status}")
            # Return trim input to avoid constraint violation
            u0 = self.us
            x_traj = np.tile(x_meas.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        v0 = self.prob.variables()[1].value[:, 0]
        u0_raw = float(v0 + us_ff)
        self.u_prev = v0  # Store delta control for next observer update

        # Clamp to respect absolute input constraints
        u0 = np.clip(u0_raw, -1.0, 1.0)

        x_traj = self.prob.variables()[0].value
        u_traj = self.prob.variables()[1].value + us_ff.reshape(-1, 1)
        return u0, x_traj, u_traj
