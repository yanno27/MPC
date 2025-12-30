import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    """
    Z-velocity MPC controller with INTEGRAL ACTION for offset-free tracking.
    
    Deliverable 5: Uses integral of tracking error to eliminate steady-state offset.
    Tuned for 50% mass change (1.0 kg → 1.5 kg).
    """
    
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        nx, nu = self.nx, self.nu
        N = self.N
        
        # ========== DELIVERABLE 5: INTEGRAL ACTION APPROACH ==========
        # Integral of tracking error for offset-free tracking
        self.error_integral = np.zeros(nx)
        
        # TUNED for 50% mass change:
        # - Ki = 0.3: Conservative gain for stability (prevents oscillations)
        # - Lower Ki → slower convergence but more stable
        self.Ki = 0.4
        
        # For logging/analysis (compatible with notebook plotting code)
        self.d_est = np.zeros(nx)  # "Disturbance estimate" = compensation
        self.x_est = np.zeros(nx)  # Kept for compatibility
        self.d_est_history = []
        self.initialized = False
        # ==============================================================
        
        Q = np.array([[100.0]])
        R = np.array([[1.0]])
        
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        A_cl = self.A + self.B @ K
        
        u_min = 40.0 - self.us[0]
        u_max = 80.0 - self.us[0]
        
        F_x = np.array([[1.0], [-1.0]])
        f_x = np.array([20.0, 20.0])
        X = Polyhedron.from_Hrep(F_x, f_x)
        
        M_u = np.array([[1.0], [-1.0]])
        m_u = np.array([u_max, -u_min])
        U = Polyhedron.from_Hrep(M_u, m_u)
        
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        Xf = self._max_invariant_set(A_cl, X.intersect(KU))
        self.Xf = Xf
        
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        x_ref_param = cp.Parameter(nx)
        
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q)
            cost += cp.quad_form(u_var[:, k], R)
        cost += cp.quad_form(x_var[:, N] - x_ref_param, Qf)
        
        constraints = []
        constraints.append(x_var[:, 0] == x0_param)
        
        for k in range(N):
            constraints.append(x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k])
        
        for k in range(N):
            constraints.append(u_var[:, k] >= u_min)
            constraints.append(u_var[:, k] <= u_max)
        
        # Terminal constraint removed (as per deliverable hint)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.x_ref_param = x_ref_param

    def _max_invariant_set(self, A_cl, X, max_iter=30):
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.concatenate((f, f)))
            O.minHrep(True)
            _ = O.Vrep
            if O == Oprev:
                converged = True
                break
            itr += 1
        if not converged:
            print(f"Warning: Terminal set did not converge after {max_iter} iterations")
        return O

    def get_u(self, x0, x_target=None, u_target=None):
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
        
        if x_target is not None and x_target.shape[0] > self.nx:
            x_target = x_target[self.x_ids]
        
        if x_target is None:
            x_ref = self.xs
        else:
            x_ref = x_target
        
        # ========== DELIVERABLE 5: INTEGRAL ACTION ==========
        if not self.initialized:
            self.error_integral = np.zeros(self.nx)
            self.initialized = True
            print(f"[Z-vel] Integral action initialized (Ki={self.Ki})")
        
        # Compute tracking error (current state minus target)
        error = x0 - x_ref

        # Accumulate error over time
        self.error_integral += self.Ts * error

        # Anti-windup: limit integral to prevent windup
        max_integral = 50.0
        self.error_integral = np.clip(self.error_integral, -max_integral, max_integral)

        # Compute disturbance estimate from integral term
        # This acts as feedforward compensation
        d_est = self.Ki * self.error_integral

        # Apply integral action: shift the reference, not the state
        # This makes MPC track a shifted reference that compensates for disturbance
        delta_x0 = (x0 - self.xs)
        delta_x_ref = (x_ref - self.xs) - d_est  # Shift reference DOWN by disturbance estimate

        # Store disturbance estimate for logging/plotting
        self.d_est = d_est.copy()
        self.d_est_history.append(self.d_est.copy())
        # ====================================================

        self.x0_param.value = delta_x0
        self.x_ref_param.value = delta_x_ref
        
        self.ocp.solve(solver=cp.CLARABEL, verbose=False)
        
        if self.ocp.status != cp.OPTIMAL:
            print(f"[Z-vel] Warning: status={self.ocp.status}")
            print(f"  delta_x0={delta_x0[0]:.3f}, d_est={d_est[0]:.3f}, integral={self.error_integral[0]:.3f}")
            u0 = np.zeros(self.nu)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
            return u0, x_traj, u_traj
        
        delta_u_opt = self.u_var.value
        delta_x_opt = self.x_var.value
        
        u0 = delta_u_opt[:, 0] + self.us
        x_traj = delta_x_opt + self.xs.reshape(-1, 1)
        u_traj = delta_u_opt + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj