import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base



class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:        
        # State and input dimensions
        nx, nu = self.nx, self.nu
        N = self.N
        
        # Tuning matrices [omega_z, gamma]
        # Increased weight for nonlinear robustness
        Q = np.diag([200.0, 50.0]) 
        R = np.array([[1.0]]) 
        
        # LQR terminal controller and cost
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        A_cl = self.A + self.B @ K
        
        # Constraints
        u_min = -20.0 - self.us[0]
        u_max = 20.0 - self.us[0]

        # State constraints (loose for roll)
        F_x = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        f_x = np.array([100, 100, 100, 100])
        X = Polyhedron.from_Hrep(F_x, f_x)
        
        # Input constraints
        M_u = np.array([[1.0], [-1.0]])
        m_u = np.array([u_max, -u_min])
        U = Polyhedron.from_Hrep(M_u, m_u)
        
        # Compute terminal invariant set
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        Xf = self._max_invariant_set(A_cl, X.intersect(KU))
        self.Xf = Xf
        
        # CVXPY variables
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        x_ref_param = cp.Parameter(nx)  # Reference state parameter

        # Cost function with reference tracking
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q) 
            cost += cp.quad_form(u_var[:, k], R)
        cost += cp.quad_form(x_var[:, N] - x_ref_param, Qf)  
        
        # Constraints
        constraints = []
        constraints.append(x_var[:, 0] == x0_param)
        
        # System dynamics
        for k in range(N):
            constraints.append(x_var[:, k + 1] == self.A @ x_var[:, k] + self.B @ u_var[:, k])
        
        # Input constraints
        for k in range(N):
            constraints.append(u_var[:, k] >= u_min)
            constraints.append(u_var[:, k] <= u_max)
        
        # Terminal constraint
        constraints.append(Xf.A @ x_var[:, N] <= Xf.b)
        
        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.x_var = x_var
        self.u_var = u_var
        self.x0_param = x0_param
        self.x_ref_param = x_ref_param # Store reference parameter

    def _max_invariant_set(self, A_cl: np.ndarray, X: Polyhedron, max_iter: int = 30) -> Polyhedron:
        """Compute maximal invariant set for autonomous system x+ = A_cl @ x"""
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

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Extract subsystem states if full state is provided
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
                
        # Reference tracking support
        if x_target is None:
            x_ref = self.xs
        else:
            if x_target.shape[0] != self.nx:
                x_target = x_target[self.x_ids]

            x_ref = x_target
                
        # Compute delta state and reference (deviation from trim)
        delta_x0 = x0 - self.xs
        delta_x_ref = x_ref - self.xs 
        
        # Set parameter and ref parameter
        self.x0_param.value = delta_x0
        self.x_ref_param.value = delta_x_ref
        
        # Solve
        self.ocp.solve(solver=cp.CLARABEL, verbose=False)
        
        # Check if solution is optimal
        if self.ocp.status != cp.OPTIMAL:
            print(f"Warning: Optimization problem status is {self.ocp.status}")
            u0 = np.zeros(self.nu)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
            return u0, x_traj, u_traj
        
        # Extract solution
        delta_u_opt = self.u_var.value
        delta_x_opt = self.x_var.value
        
        # Add back trim
        u0 = delta_u_opt[:, 0] + self.us
        x_traj = delta_x_opt + self.xs.reshape(-1, 1)
        u_traj = delta_u_opt + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj