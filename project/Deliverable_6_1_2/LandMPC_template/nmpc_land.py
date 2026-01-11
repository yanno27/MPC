import numpy as np
import casadi as ca
from typing import Tuple
from scipy.linalg import solve_continuous_are


class NmpcCtrl:
    """
    Nonlinear MPC controller for rocket landing.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = nmpc.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    """

    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray):
        """
        Args:
            rocket: Rocket object with f_symbolic method
            H: Prediction horizon in seconds
            xs: Steady-state target (12,)
            us: Steady-state input (4,)
        """
        self.rocket = rocket
        self.Ts = rocket.Ts
        self.H = H
        self.N = int(H / self.Ts)
        
        self.xs = xs
        self.us = us
        
        # Symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]
        
        # Compute terminal cost matrix
        self.P = self._compute_terminal_cost()
        
        # Setup controller
        self._setup_controller()

    def _compute_terminal_cost(self) -> np.ndarray:
        """
        Compute terminal cost matrix P by solving DARE for linearized system.
        """
        # Linearize around target
        A, B = self.rocket.linearize(self.xs, self.us)
        
        # Discretize
        from scipy.signal import cont2discrete
        Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(12), np.zeros((12, 4))), self.Ts)
        
        # Stage cost matrices (same as in optimization)
        Q = np.diag([10, 10, 10,        # angular velocities
                     100, 100, 20,      # angles (alpha, beta, gamma)
                     20, 20, 100,       # velocities
                     200, 200, 200])    # positions
        R = np.diag([1, 1, 0.1, 1])     # inputs
        
        # Solve DARE for terminal cost
        try:
            P = solve_continuous_are(Ad, Bd, Q, R)
        except:
            print("Warning: DARE failed, using Q as terminal cost")
            P = Q * 10  # Scale up Q for terminal cost
            
        return P

    def _setup_controller(self) -> None:
        """Setup the NMPC optimization problem using CasADi Opti."""
        
        # Create Opti object
        opti = ca.Opti()
        
        # Decision variables
        X = opti.variable(12, self.N + 1)  # States: [w(3), phi(3), v(3), p(3)]
        U = opti.variable(4, self.N)        # Inputs: [delta1, delta2, Pavg, Pdiff]
        
        # Parameter: initial state
        x0_param = opti.parameter(12, 1)
        
        # Stage cost matrices
        Q = np.diag([10, 10, 10,        # angular velocities
                     100, 100, 20,      # angles (alpha, beta, gamma)
                     20, 20, 100,       # velocities (vx, vy, vz)
                     200, 200, 200])    # positions (x, y, z)
        R = np.diag([1, 1, 0.1, 1])     # inputs
        
        # Objective function
        obj = 0
        
        # Stage costs
        for k in range(self.N):
            dx = X[:, k] - self.xs
            du = U[:, k] - self.us
            obj += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])
        
        # Terminal cost
        dx_N = X[:, self.N] - self.xs
        obj += ca.mtimes([dx_N.T, self.P, dx_N])
        
        opti.minimize(obj)
        
        # Dynamics constraints (RK4 integration)
        for k in range(self.N):
            x_next = self._rk4(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)
        
        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0_param)
        
        # State constraints
        for k in range(self.N + 1):
            # |beta| <= 80 degrees (beta is index 4 in [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z])
            opti.subject_to(X[4, k] >= -80 * np.pi / 180)
            opti.subject_to(X[4, k] <= 80 * np.pi / 180)
            
            # z >= 0 (z is index 11)
            opti.subject_to(X[11, k] >= 0)
        
        # Input constraints
        for k in range(self.N):
            # delta1, delta2: [-0.26, 0.26] rad (±15 degrees)
            opti.subject_to(opti.bounded(-0.26, U[0, k], 0.26))
            opti.subject_to(opti.bounded(-0.26, U[1, k], 0.26))
            
            # Pavg: [40, 80] %
            opti.subject_to(opti.bounded(40, U[2, k], 80))
            
            # Pdiff: [-20, 20] %
            opti.subject_to(opti.bounded(-20, U[3, k], 20))
        
        # Solver options
        opts = {
            'expand': True,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)
        
        # Store for later use
        self.opti = opti
        self.X = X
        self.U = U
        self.x0_param = x0_param
        
        # Initialize with steady state
        opti.set_initial(X, np.tile(self.xs.reshape(-1, 1), (1, self.N + 1)))
        opti.set_initial(U, np.tile(self.us.reshape(-1, 1), (1, self.N)))

    def _rk4(self, x: ca.SX, u: ca.SX) -> ca.SX:
        """
        RK4 integration step for dynamics.
        """
        k1 = self.f(x, u)
        k2 = self.f(x + self.Ts / 2 * k1, u)
        k3 = self.f(x + self.Ts / 2 * k2, u)
        k4 = self.f(x + self.Ts * k3, u)
        return x + self.Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve NMPC optimization and return optimal control.
        
        Args:
            t0: Current time
            x0: Current state (12,)
            
        Returns:
            u0: First control input to apply (4,)
            x_ol: Optimal state trajectory (12, N+1)
            u_ol: Optimal input trajectory (4, N)
            t_ol: Time vector (N+1,)
        """
        
        # Set initial condition parameter
        self.opti.set_value(self.x0_param, x0)
        
        # Solve optimization problem
        try:
            sol = self.opti.solve()
            
            # Extract optimal solution
            x_ol = sol.value(self.X)
            u_ol = sol.value(self.U)
            
            # Warm start next iteration with shifted solution
            if self.N > 1:
                x_init = np.hstack([x_ol[:, 1:], x_ol[:, -1:]])
                u_init = np.hstack([u_ol[:, 1:], u_ol[:, -1:]])
            else:
                x_init = x_ol
                u_init = u_ol
            self.opti.set_initial(self.X, x_init)
            self.opti.set_initial(self.U, u_init)
            
        except RuntimeError as e:
            print(f"NMPC solver failed at t={t0:.2f}s: {e}")
            print("Using debug values from failed solve")
            # Use debug values if optimization fails
            x_ol = self.opti.debug.value(self.X)
            u_ol = self.opti.debug.value(self.U)
        
        # Extract first control input
        u0 = u_ol[:, 0]
        
        # Generate time vector
        t_ol = t0 + np.arange(self.N + 1) * self.Ts
        
        return u0, x_ol, u_ol, t_ol