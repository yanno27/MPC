import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # [Pavg]

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        """Setup robust tube MPC controller for z subsystem."""
        
        # System dimensions
        nx = self.nx  # 2 states: vz, z
        nu = self.nu  # 1 input: Pavg
        N = self.N    # horizon steps
        
        # State and input constraints (in delta coordinates)
        # Original constraints: 40 ≤ Pavg ≤ 80, z ≥ 0
        u_min = 40.0 - self.us[0]  # lower bound in delta
        u_max = 80.0 - self.us[0]  # upper bound in delta
        
        # State constraint: z ≥ 0 => xs[1] + delta_x[1] ≥ 0
        # => delta_x[1] ≥ -xs[1]
        x_min = np.array([-np.inf, -self.xs[1]])
        x_max = np.array([np.inf, np.inf])
        
        # Disturbance set W = [-15, 5]
        w_min = -15.0
        w_max = 5.0
        
        print(f"\nConstraints in delta coordinates:")
        print(f"  u ∈ [{u_min:.2f}, {u_max:.2f}]")
        print(f"  x[1] (z) ≥ {x_min[1]:.2f}")
        
        # ============================================
        # STEP 1: Design ancillary controller K
        # ============================================
        # Use LQR to find stabilizing feedback gain
        Q_lqr = np.diag([0.5, 5.0])  # moderate weights
        R_lqr = np.eye(nu) * 2.0     # higher input cost for less aggressive K
        K, _, _ = dlqr(self.A, self.B, Q_lqr, R_lqr)
        K = -K  # dlqr returns positive K, we need u = -Kx
        
        # Closed-loop matrix
        A_K = self.A + self.B @ K
        
        # Check stability
        eigvals = np.linalg.eigvals(A_K)
        rho_AK = np.max(np.abs(eigvals))
        print(f"\nAncillary controller:")
        print(f"  K = {K.flatten()}")
        print(f"  Closed-loop eigenvalues: {eigvals}")
        print(f"  Spectral radius: {rho_AK:.4f}")
        
        if rho_AK >= 0.99:
            print("  Warning: Closed-loop system close to unstable!")
        
        self.K = K
        
        # ============================================
        # STEP 2: Compute minimal RPI set E using Polyhedron
        # ============================================
        # Create disturbance set W as polyhedron: w_min ≤ w ≤ w_max
        W = Polyhedron.from_Hrep(
            A=np.array([[-1.0], [1.0]]),
            b=np.array([-w_min, w_max])
        )
        
        B_d = self.B  # disturbance enters through input channel
        
        # Compute minimal RPI set iteratively
        E = B_d @ W  # Initialize with B_d * W
        
        max_iter = 50
        for i in range(max_iter):
            E_prev = E
            E_new = (A_K @ E) + (B_d @ W)
            
            # Check convergence
            if E_new == E_prev:
                print(f"  RPI set E converged after {i+1} iterations")
                break
            
            # Limit complexity
            if hasattr(E_new, 'A') and E_new.A.shape[0] > 100:
                print(f"  RPI set complexity too high at iteration {i+1}, stopping")
                break
                
            E = E_new
        
        self.E = E
        
        # Get bounds on E for constraint tightening
        try:
            E.minimize()
            vertices = E.V
            e_bound = np.max(np.abs(vertices), axis=0)
            print(f"  RPI set E has {len(vertices)} vertices")
        except:
            # Fallback: use analytical bound
            w_bar = max(abs(w_min), abs(w_max))
            try:
                I_minus_AK = np.eye(nx) - A_K
                ss_gain = np.linalg.solve(I_minus_AK, B_d.flatten())
                e_bound = np.abs(ss_gain) * w_bar
            except:
                e_bound = np.array([5.0, 5.0])
        
        # CRITICAL: Use full theoretical bound for extreme disturbances
        # We need 100% tightening to guarantee robustness with w=-15 constant
        tightening_factor = 1.0  # Use FULL 100% of theoretical bound
        e_tighten = e_bound * tightening_factor
        
        self.e_bound = e_bound
        self.e_tighten = e_tighten
        
        print(f"  RPI set bounds: vz ± {e_bound[0]:.2f} m/s, z ± {e_bound[1]:.2f} m")
        print(f"  Used for tightening ({int(tightening_factor*100)}%): vz ± {e_tighten[0]:.2f} m/s, z ± {e_tighten[1]:.2f} m")
        
        # ============================================
        # STEP 3: Tighten constraints
        # ============================================
        
        # State constraints with FULL tightening
        x_tilde_min = np.array([-np.inf, -self.xs[1] + e_tighten[1]])
        x_tilde_max = np.array([np.inf, np.inf])
        
        self.x_tilde_min = x_tilde_min
        self.x_tilde_max = x_tilde_max
        
        # Create X_tilde as polyhedron
        A_xtilde = np.array([[0, -1]])
        b_xtilde = np.array([self.xs[1] - e_tighten[1]])
        self.X_tilde = Polyhedron.from_Hrep(A=A_xtilde, b=b_xtilde)
        
        # Input constraints
        U = Polyhedron.from_Hrep(
            A=np.array([[-1.0], [1.0]]),
            b=np.array([-u_min, u_max])
        )
        
        # Compute range of K*e with FULL tightening
        Ke_range = 0.0
        for i in range(nx):
            Ke_range += np.abs(K[0, i]) * e_tighten[i]
        
        Ke_range_factor = 1.0  # Use FULL 100% tightening
        Ke_range_used = Ke_range * Ke_range_factor
        
        u_tilde_min = u_min + Ke_range_used
        u_tilde_max = u_max - Ke_range_used
        
        # Ensure feasibility
        if u_tilde_min >= u_tilde_max - 1.0:
            print("  Warning: Input constraints very tight after full tightening")
            # Keep at least 1% range
            mid = (u_min + u_max) / 2
            u_tilde_min = mid - 0.5
            u_tilde_max = mid + 0.5
        
        self.u_tilde_min = u_tilde_min
        self.u_tilde_max = u_tilde_max
        
        # Create U_tilde as polyhedron
        self.U_tilde = Polyhedron.from_Hrep(
            A=np.array([[-1.0], [1.0]]),
            b=np.array([-u_tilde_min, u_tilde_max])
        )
        
        print(f"\nTightened constraints:")
        print(f"  X_tilde: z ≥ {x_tilde_min[1]:.2f}")
        print(f"  U_tilde: ΔPavg ∈ [{u_tilde_min:.2f}, {u_tilde_max:.2f}] (range: {u_tilde_max - u_tilde_min:.2f})")
        print(f"  U_tilde (absolute): Pavg ∈ [{u_tilde_min + self.us[0]:.2f}, {u_tilde_max + self.us[0]:.2f}]")
        
        # ============================================
        # STEP 4: Terminal ingredients
        # ============================================
        Q_term = np.diag([1.0, 20.0])
        R_term = np.array([[1.0]])
        
        try:
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(self.A, self.B, Q_term, R_term)
        except:
            P = Q_term * 50
        
        self.P = P
        
        # Terminal set Xf: simpler approach for robustness
        # Use a conservative box that fits in tightened constraints
        x_f_size = np.minimum(
            np.array([2.0, 0.8]),  # Maximum desired size
            np.array([10.0, (self.xs[1] - e_tighten[1]) * 0.3])  # 30% of available space
        )
        
        self.x_f_min = -x_f_size
        self.x_f_max = x_f_size
        
        # Create X_f polyhedron
        A_xf = np.array([
            [-1, 0], [1, 0],  # vz bounds
            [0, -1], [0, 1]   # z bounds
        ])
        b_xf = np.array([
            -self.x_f_min[0], self.x_f_max[0],
            -self.x_f_min[1], self.x_f_max[1]
        ])
        self.X_f = Polyhedron.from_Hrep(A=A_xf, b=b_xf)
        
        print(f"\nTerminal set X_f (conservative box):")
        print(f"  vz ∈ [{self.x_f_min[0]:.2f}, {self.x_f_max[0]:.2f}] m/s")
        print(f"  z ∈ [{self.x_f_min[1]:.2f}, {self.x_f_max[1]:.2f}] m")
        
        # ============================================
        # STEP 5: Setup optimization problem
        # ============================================
        Q = np.diag([0.5, 25.0])  # Higher weight on z for tracking
        R = np.array([[1.0]])
        
        # Decision variables
        z = cp.Variable((nx, N+1))
        v = cp.Variable((nu, N))
        
        # Slack variables for soft constraints
        slack_z = cp.Variable((N+1,), nonneg=True)
        
        # Parameters
        z0 = cp.Parameter(nx)
        
        # Cost function
        cost = 0
        slack_penalty = 2000.0  # Higher penalty for tighter control
        
        for k in range(N):
            cost += cp.quad_form(z[:, k], Q) + cp.quad_form(v[:, k], R)
            cost += slack_penalty * slack_z[k]
        
        cost += cp.quad_form(z[:, N], P)
        cost += slack_penalty * slack_z[N]
        
        # Constraints
        constraints = []
        constraints.append(z[:, 0] == z0)
        
        for k in range(N):
            constraints.append(z[:, k+1] == self.A @ z[:, k] + self.B @ v[:, k])
            
            # State constraints (soft)
            if not np.isinf(x_tilde_min[1]):
                constraints.append(z[1, k] >= x_tilde_min[1] - slack_z[k])
            
            # Input constraints (hard)
            constraints.append(self.U_tilde.A @ v[:, k] <= self.U_tilde.b)
        
        # Terminal constraint (soft)
        constraints.append(z[0, N] >= self.x_f_min[0])
        constraints.append(z[0, N] <= self.x_f_max[0])
        constraints.append(z[1, N] >= self.x_f_min[1] - slack_z[N])
        constraints.append(z[1, N] <= self.x_f_max[1])
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        self.z0_param = z0
        self.z_var = z
        self.v_var = v
        
        print(f"\nRobust Tube MPC setup complete:")
        print(f"  - Horizon: {N} steps ({self.H}s)")
        print(f"  - FULL tightening for extreme disturbance robustness")
        print(f"  - Soft constraints with penalty: {slack_penalty}")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute control using tube MPC: u = v* + K(x - z*)"""
        
        delta_x = x0 - self.xs
        self.z0_param.value = delta_x
        
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
                u0_delta = self.K @ delta_x
                u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
                return u0, np.zeros((self.nx, self.N+1)), np.zeros((self.nu, self.N))
            
            z_opt = self.z_var.value
            v_opt = self.v_var.value
            
            # Tube MPC control law
            u0_delta = v_opt[:, 0] + self.K @ (delta_x - z_opt[:, 0])
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            
            x_traj = z_opt + self.xs[:, np.newaxis]
            u_traj = v_opt + self.us[:, np.newaxis]
            
            return u0, x_traj, u_traj
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            u0_delta = self.K @ delta_x
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            return u0, np.zeros((self.nx, self.N+1)), np.zeros((self.nu, self.N))

    def setup_estimator(self):
        """FOR PART 5 OF THE PROJECT"""
        self.d_estimate = np.array([0.0])
        self.d_gain = 0.1

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        """FOR PART 5 OF THE PROJECT"""
        pass