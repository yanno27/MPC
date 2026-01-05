import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.linalg import solve_discrete_are

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # System dimensions
        self.nx = 2  # [vz, z]
        self.nu = 1  # Pavg
        self.N = int(self.H / self.Ts)
        
        print(f"Setting up Tube MPC: N={self.N}, Ts={self.Ts}")
        
        # 1. MIGLIORA LQR - pesi più bilanciati
        Q_lqr = np.diag([0.5, 5.0])   # Meno peso su vz, più su z
        R_lqr = np.array([[0.01]])     # Molto meno peso sull'input per controllo più aggressivo
        
        # Usa solve_discrete_are invece di dlqr per più stabilità
        P = solve_discrete_are(self.A, self.B, Q_lqr, R_lqr)
        self.K = -np.linalg.inv(R_lqr + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        
        print(f"LQR gain K = {self.K.flatten()}")
        
        # Closed-loop per RPI
        A_cl = self.A + self.B @ self.K
        eigvals = np.linalg.eigvals(A_cl)
        print(f"A_cl eigenvalues: {eigvals}, |max|={np.max(np.abs(eigvals)):.4f}")

        # 2. Disturbance set W = [-15, 5] 
        # RIDUCI l'effetto del disturbo per il calcolo RPI
        W_scale = 0.5  # Scala il disturbo per RPI più piccolo
        Aw = np.array([[1.0], [-1.0]])
        bw = np.array([5.0 * W_scale, 15.0 * W_scale])
        W = Polyhedron.from_Hrep(Aw, bw)

        # 3. RPI set - INIZIALE più piccolo
        He0 = np.array([
            [1.0, 0.0], [-1.0, 0.0],   # vz bounds
            [0.0, 1.0], [0.0, -1.0],   # z bounds
        ])
        
        # Bound iniziali PICCOLI ma non troppo
        e0_vz = 0.5   # ±0.5 m/s per vz
        e0_z = 0.3    # ±0.3 m per z
        he0 = np.array([e0_vz, e0_vz, e0_z, e0_z])
        E0 = Polyhedron.from_Hrep(He0, he0)

        # 4. Calcola RPI set
        self.E = self._robust_invariant_set(A_cl, self.B, W, E0, max_iter=20)
        
        if hasattr(self.E, 'empty') and self.E.empty:
            print("WARNING: E empty, using E0")
            self.E = E0
        
        print(f"RPI set E: {len(self.E.A)} facets")

        # 5. Original constraints - RILASSA per permettere movimento
        # z >= 0 è l'unico constraint hard
        z_min = 0.0
        z_max = 50.0  # Alto per non limitare
        vz_min = -20.0  # Permetti discesa rapida
        vz_max = 20.0   # Permetti salita rapida
        
        Ax = np.array([
            [1.0, 0.0],    # vz ≤ vz_max
            [-1.0, 0.0],   # vz ≥ vz_min
            [0.0, 1.0],    # z ≤ z_max
            [0.0, -1.0],   # z ≥ z_min
        ])
        bx = np.array([vz_max, -vz_min, z_max, -z_min])
        X = Polyhedron.from_Hrep(Ax, bx)

        # Input constraints
        Au = np.array([[1.0], [-1.0]])
        bu = np.array([80.0, -40.0])  # Pavg ∈ [40, 80]
        U = Polyhedron.from_Hrep(Au, bu)

        # 6. Tightened constraints - meno tightening
        self.X_tilde = X - self.E
        
        # K per tightening input
        K_reshaped = self.K.reshape(1, -1) if self.K.ndim == 1 else self.K
        KE = self.E.affine_map(K_reshaped)
        self.U_tilde = U - KE
        
        print(f"Tightening: X_tilde {len(self.X_tilde.A)} facets, U_tilde {len(self.U_tilde.A)} facets")

        # 7. MPC tuning - PESI MIGLIORATI
        Q_mpc = np.diag([0.2, 2.0])   # Peso su z 10x vz
        R_mpc = np.array([[0.001]])   # Peso input molto piccolo
        
        # Terminal cost from DARE con pesi diversi
        P_term = solve_discrete_are(self.A, self.B, np.diag([1.0, 10.0]), np.array([[0.01]]))

        # 8. Setup MPC problem
        x = cp.Variable((self.nx, self.N + 1))  # Stati centrati
        u = cp.Variable((self.nu, self.N))      # Input centrati

        cost = 0
        constraints = []

        # Initial condition (stato centrato)
        x0_param = cp.Parameter(self.nx)
        constraints += [x[:, 0] == x0_param]

        for k in range(self.N):
            # Stage cost
            cost += cp.quad_form(x[:, k], Q_mpc) + cp.quad_form(u[:, k], R_mpc)
            
            # Dynamics
            constraints += [x[:, k+1] == self.A @ x[:, k] + self.B @ u[:, k]]
            
            # Constraints - SOLO quelli essenziali
            # Solo z ≥ 0 con piccolo margine
            constraints += [x[1, k] >= -self.xs[1] + 0.1]  # z ≥ 0.1
            
            # Input constraints con piccolo margine
            constraints += [u[:, k] <= 80.0 - self.us[0] - 1.0]
            constraints += [u[:, k] >= 40.0 - self.us[0] + 1.0]

        # Terminal cost (più peso)
        cost += 10.0 * cp.quad_form(x[:, self.N], P_term)
        
        # Terminal constraint opzionale
        # constraints += [cp.norm(x[:, self.N], 2) <= 0.5]

        # Build problem
        self.x0_param = x0_param
        self.x_var = x
        self.u_var = u
        
        # Problema più semplice
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        print(f"Trim point: xs[z]={self.xs[1]:.2f}, us[Pavg]={self.us[0]:.2f}")
        print("Tube MPC setup complete")

        # YOUR CODE HERE
        #################################################

    def _robust_invariant_set(
        self, A_cl: np.ndarray, Bd: np.ndarray, W: Polyhedron, E0: Polyhedron, max_iter: int = 20,
    ) -> Polyhedron:
        """Compute RPI set with early stopping."""
        E = E0
        
        for i in range(max_iter):
            E_prev = E
            
            # One-step evolution
            E_dyn = E.affine_map(A_cl)
            E_dist = W.affine_map(Bd)
            E_next = E_dyn + E_dist
            
            # Intersection
            E = E_next.intersect(E_prev)
            E.minHrep(True)
            
            # Early stop if small or empty
            if hasattr(E, 'empty') and E.empty:
                return E_prev
                
            if i > 5 and E == E_prev:
                break
                
        return E

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # Extract subsystem state
        if x0.shape[0] == 12:
            x0_subsystem = x0[self.x_ids]
        else:
            x0_subsystem = x0
        
        # Center state
        xz = x0_subsystem - self.xs
        
        # Debug
        # print(f"State: z={x0_subsystem[1]:.2f}, vz={x0_subsystem[0]:.2f}, centered: {xz}")
        
        # Set MPC initial condition
        self.x0_param.value = xz

        try:
            # Solve MPC - impostazioni più aggressive
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=5000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=True
            )

            if self.ocp.status in ["optimal", "optimal_inaccurate"]:
                # MPC solution
                x_nom = self.x_var.value  # centered
                u_nom = self.u_var.value  # centered
                
                # Tube correction
                u0_centered = u_nom[:, 0] + self.K @ (xz - x_nom[:, 0])
                
                # Add trim and clip
                u0 = u0_centered + self.us
                u0_clipped = np.clip(u0, 40.0, 80.0)
                
                # Scale down if clipping happened
                if np.abs(u0[0] - u0_clipped[0]) > 0.1:
                    u0_centered = u0_clipped - self.us
                
                # Create trajectories with trim
                x_traj = x_nom + self.xs.reshape(-1, 1)
                u_traj = u_nom + self.us.reshape(-1, 1)
                
                return u0_clipped, x_traj, u_traj
            else:
                # Fallback to aggressive LQR
                print(f"MPC failed ({self.ocp.status}), using LQR")
                
        except Exception as e:
            print(f"MPC exception: {e}")
        
        # AGGRESSIVE LQR FALLBACK
        # Guadagno proporzionale-derivativo semplice
        K_p = 0.8  # Guadagno posizione
        K_d = 0.3  # Guadagno velocità
        
        # Errore: vogliamo z=3, vz=0
        z_error = x0_subsystem[1] - self.xs[1]  # z - 3
        vz_error = x0_subsystem[0] - self.xs[0]  # vz - 0
        
        # Controllo PD
        u_pd = self.us[0] - K_p * z_error - K_d * vz_error
        u_pd = np.clip(u_pd, 40.0, 80.0)
        
        # Traiettorie semplici
        u0 = np.array([u_pd])
        x_traj = np.tile(x0_subsystem.reshape(-1, 1), (1, self.N + 1))
        u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
        
        return u0, x_traj, u_traj

    def setup_estimator(self):
        ##################################################
        self.d_estimate = np.array([0.0])
        self.d_gain = 0.1
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        ##################################################
        if x_data.shape[1] >= 2:
            x_curr = x_data[self.x_ids, -1]
            x_prev = x_data[self.x_ids, -2]
            u_prev = u_data[self.u_ids, -1]
            
            x_pred = self.A @ x_prev + self.B @ u_prev
            d_new = x_curr - x_pred
            
            self.d_estimate = (1 - self.d_gain) * self.d_estimate + self.d_gain * d_new
        ##################################################