import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base



class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # State and input dimensions
        nx, nu = self.nx, self.nu
        N = self.N
        

        Q = np.diag([50.0, 200.0])  # Penalize roll angle much more than angular velocity 3.

        R = np.array([[1.0]])  # Input cost (differential throttle)
        
        # Compute LQR terminal controller and cost
        K, Qf, _ = dlqr(self.A, self.B, Q, R)
        K = -K
        A_cl = self.A + self.B @ K
        
        # Define constraints
        # Roll angle gamma is unrestricted but we constrain differential throttle
        # u_trim = us[3] (Pdiff_trim)
        # Pdiff in [-20, 20], so delta_u in [-20-us[3], 20-us[3]]
        u_min = -20.0 - self.us[0]
        u_max = 20.0 - self.us[0]
        
        # Define state constraints (no explicit state constraints for roll)
        # We'll still define sets for terminal set computation
        x_min = np.array([-np.inf, -np.inf])
        x_max = np.array([np.inf, np.inf])
        
        # Create constraint polyhedra
        # Input constraints: u in [u_min, u_max]
        M_u = np.array([[1.0], [-1.0]])
        m_u = np.array([u_max, -u_min])
        U = Polyhedron.from_Hrep(M_u, m_u)
        
        # State constraints (very loose for roll)
        F_x = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        f_x = np.array([100, 100, 100, 100])  # Large bounds
        X = Polyhedron.from_Hrep(F_x, f_x)
        
        # Compute terminal invariant set
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        Xf = self._max_invariant_set(A_cl, X.intersect(KU))
        self.Xf = Xf  # Store for plotting
        
        # Define CVXPY variables
        x_var = cp.Variable((nx, N + 1))
        u_var = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)
        # ===== ADDED FOR DELIVERABLE 3.2: Reference tracking =====
        x_ref_param = cp.Parameter(nx)  # Reference state parameter
        # ==========================================================
        
        # ===== MODIFIED FOR DELIVERABLE 3.2: Track reference instead of origin =====
        # Cost function with reference tracking
        cost = 0
        for k in range(N):
            cost += cp.quad_form(x_var[:, k] - x_ref_param, Q)  # Changed: x -> (x - x_ref)
            cost += cp.quad_form(u_var[:, k], R)
        cost += cp.quad_form(x_var[:, N] - x_ref_param, Qf)    # Changed: x -> (x - x_ref)
        # ============================================================================
        
        # Constraints
        constraints = []
        
        # Initial condition
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
        # ===== ADDED FOR DELIVERABLE 3.2: Store reference parameter =====
        self.x_ref_param = x_ref_param
        # ================================================================

        # YOUR CODE HERE
        #################################################

    def _max_invariant_set(self, A_cl: np.ndarray, X: Polyhedron, max_iter: int = 30) -> Polyhedron:
        """Compute maximal invariant set for autonomous system x+ = A_cl @ x"""
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.concatenate((f, f)))
            O.minHrep(True)
            _ = O.Vrep  # Force computation of V-rep for robustness
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
        #################################################
        # YOUR CODE HERE
        
        # Extract subsystem states if full state is provided
        if x0.shape[0] > self.nx:
            x0 = x0[self.x_ids]
        
        # ===== ADDED FOR DELIVERABLE 3.2: Reference tracking support =====
        # Extract reference if provided (handle full state vector)
        if x_target is not None and x_target.shape[0] > self.nx:
            x_target = x_target[self.x_ids]
        
        # Default reference is trim (zero velocity/angle) for stabilization
        # When x_target is provided, track that reference instead
        if x_target is None:
            x_ref = self.xs  # Stabilization mode (same as Deliverable 3.1)
        else:
            x_ref = x_target  # Tracking mode (new in Deliverable 3.2)
        # ==================================================================
        
        # Compute delta state and reference (deviation from trim)
        delta_x0 = x0 - self.xs
        # ===== ADDED FOR DELIVERABLE 3.2 =====
        delta_x_ref = x_ref - self.xs  # Reference in delta coordinates
        # ======================================
        
        # Set parameter values
        self.x0_param.value = delta_x0
        # ===== ADDED FOR DELIVERABLE 3.2 =====
        self.x_ref_param.value = delta_x_ref  # Set reference parameter
        # ======================================
        
        # Solve optimization problem
        self.ocp.solve(solver=cp.CLARABEL, verbose=False)
        
        # Check if solution is optimal
        if self.ocp.status != cp.OPTIMAL:
            print(f"Warning: Optimization problem status is {self.ocp.status}")
            # Return zero control if infeasible
            u0 = np.zeros(self.nu)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.zeros((self.nu, self.N))
            return u0, x_traj, u_traj
        
        # Extract solution
        delta_u_opt = self.u_var.value
        delta_x_opt = self.x_var.value
        
        # First control input (add back trim)
        u0 = delta_u_opt[:, 0] + self.us
        
        # State and input trajectories (add back trim)
        x_traj = delta_x_opt + self.xs.reshape(-1, 1)
        u_traj = delta_u_opt + self.us.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj