import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9]) 
    u_ids: np.ndarray = np.array([1])  

    def _setup_controller(self) -> None:
        # State and input dimensions
        nx, nu = self.nx, self.nu
        N = self.N
        
        # Tuning matrices [omega_y, beta, vx]
        Q = np.diag([0.1, 5.0, 1.0, 20.0])
        R = np.array([[0.1]])
        
        # Terminal cost from DARE
        P_term = solve_discrete_are(self.A, self.B, Q, R)
        
        # Constraints
        beta_max = 0.1745  # 10 degrees
        u_min = -0.2618 - self.us[0]  # -15 degrees
        u_max = 0.2618 - self.us[0]   # +15 degrees
        
        # Soft constraint weights
        slack_weight = 1000.0 
        
        # CVXPY variables
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        x_ref_param = cp.Parameter(nx)
        slack_beta = cp.Variable((N, 1), nonneg=True)
        
        # Cost function
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q)
            cost += cp.quad_form(u_var[:, k], R)
            cost += slack_weight * cp.sum_squares(slack_beta[k])
        
        # Terminal cost
        cost += cp.quad_form(x_var[:, N] - x_ref_param, P_term)
        
        # Constraints
        constraints = []
        constraints.append(x_var[:, 0] == x0_param)
        
        # System ynamics
        for k in range(N):
            constraints.append(x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k])
        
        # State constraints 
        for k in range(N):
            constraints.append(x_var[1, k] <= beta_max + slack_beta[k])
            constraints.append(-x_var[1, k] <= beta_max + slack_beta[k])
        
        # Input constraints
        for k in range(N):
            constraints.append(u_var[:, k] >= u_min)
            constraints.append(u_var[:, k] <= u_max)
        
        # Optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.x_ref_param = x_ref_param
        self.slack_beta = slack_beta

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extract subsystem states if full state provided
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
        
        # Reference tracking support
        if x_target is None:
            x_ref = self.xs
        else:
            if x_target.shape[0] != self.nx:
                x_target = x_target[self.x_ids]
            x_ref = x_target
        
        # Compute deviations from trim
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
                # Extract solution
                delta_u_opt = self.u_var.value
                delta_x_opt = self.x_var.value
                
                # Add back trim
                u0 = delta_u_opt[:, 0] + self.us
                x_traj = delta_x_opt + self.xs.reshape(-1, 1)
                u_traj = delta_u_opt + self.us.reshape(-1, 1)
                
                # Clip input to hard limits
                u0 = np.clip(u0, -0.2618, 0.2618)
                
                return u0, x_traj, u_traj
            else:
                print(f"MPC_x failed ({self.ocp.status}), using zero input")
                
        except Exception as e:
            print(f"MPC_x exception: {e}")
        
        # Fallback
        u0 = np.zeros(self.nu)
        x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
        u_traj = np.zeros((self.nu, self.N))
        
        return u0, x_traj, u_traj