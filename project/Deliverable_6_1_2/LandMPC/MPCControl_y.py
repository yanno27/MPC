import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])  # ω_x, α, v_y, y
    u_ids: np.ndarray = np.array([0])  # δ₁

    def _setup_controller(self) -> None:
        """Setup MPC for y-position control with soft constraints."""
        
        # Dimensions
        nx, nu = self.nx, self.nu
        N = self.N
        
        print(f"Setting up MPC_y: N={N}, Ts={self.Ts}")
        
        # --- Tuning matrices ---
        # States: [ω_x, α, v_y, y]
        Q = np.diag([0.1, 5.0, 1.0, 20.0])
        R = np.array([[0.1]])
        
        # Terminal cost from DARE
        P_term = solve_discrete_are(self.A, self.B, Q, R)
        
        # --- Constraints ---
        # Angle constraint: |α| ≤ 10° = 0.1745 rad
        alpha_max = 0.1745
        
        # Input constraint: |δ₁| ≤ 15° = 0.2618 rad
        delta_max = 0.2618
        u_min = -delta_max - self.us[0]
        u_max = delta_max - self.us[0]
        
        # Soft constraint weight
        slack_weight = 1000.0
        
        # --- CVXPY Variables ---
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        x_ref_param = cp.Parameter(nx)
        
        # Slack variables for soft state constraints
        slack_alpha = cp.Variable((N, 1), nonneg=True)
        
        # --- Cost Function ---
        cost = 0
        
        # Stage costs
        for k in range(N):
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q)
            cost += cp.quad_form(u_var[:, k], R)
            cost += slack_weight * cp.sum_squares(slack_alpha[k])
        
        # Terminal cost
        cost += cp.quad_form(x_var[:, N] - x_ref_param, P_term)
        
        # --- Constraints ---
        constraints = []
        
        # Initial condition
        constraints.append(x_var[:, 0] == x0_param)
        
        # Dynamics
        for k in range(N):
            constraints.append(
                x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k]
            )
        
        # State constraints (soft on α)
        for k in range(N):
            constraints.append(x_var[1, k] <= alpha_max + slack_alpha[k])
            constraints.append(-x_var[1, k] <= alpha_max + slack_alpha[k])
        
        # Input constraints (hard)
        for k in range(N):
            constraints.append(u_var[:, k] >= u_min)
            constraints.append(u_var[:, k] <= u_max)
        
        # --- Build Problem ---
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        # Store variables
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.x_ref_param = x_ref_param
        self.slack_alpha = slack_alpha
        
        print("MPC_y setup complete")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get control input for current state."""
        
        # Extract subsystem states
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
        
        # Set reference
        if x_target is None:
            x_ref = self.xs
        else:
            if x_target.shape[0] != self.nx:
                x_target = x_target[self.x_ids]
            x_ref = x_target
        
        # Compute deviations
        delta_x0 = x0 - self.xs
        delta_x_ref = x_ref - self.xs
        
        # Set parameters
        self.x0_param.value = delta_x0
        self.x_ref_param.value = delta_x_ref
        
        # Solve
        try:
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=4000
            )
            
            if self.ocp.status in ["optimal", "optimal_inaccurate"]:
                delta_u_opt = self.u_var.value
                delta_x_opt = self.x_var.value
                
                u0 = delta_u_opt[:, 0] + self.us
                x_traj = delta_x_opt + self.xs.reshape(-1, 1)
                u_traj = delta_u_opt + self.us.reshape(-1, 1)
                
                # Clip to hard limits
                u0 = np.clip(u0, -0.2618, 0.2618)
                
                return u0, x_traj, u_traj
            else:
                print(f"MPC_y failed ({self.ocp.status}), using zero input")
                
        except Exception as e:
            print(f"MPC_y exception: {e}")
        
        # Fallback
        u0 = np.zeros(self.nu)
        x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
        u_traj = np.zeros((self.nu, self.N))
        
        return u0, x_traj, u_traj