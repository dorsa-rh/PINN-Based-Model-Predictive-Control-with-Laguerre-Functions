import logging
import time

import numpy as np
import tensorflow as tf

from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

from itertools import combinations

import control
from polytope import Polytope
from pypolycontain.operations import minkowski_sum

from scipy.integrate import solve_ivp

class MPC2:
    """
    Class used to represent a Model Predictive Controller.
    """

    def __init__(self, plant, model, x0,u0, u_ub, u_lb, Gx, gx, Gu, gu, Gw, gw, t_sample=0.1, H=10, w=0.01, 
                 Q=tf.eye(1, dtype=tf.float64), R=tf.eye(1, dtype=tf.float64)):
        self.plant = plant
        self.model = model
        self.H = H
        self.t_sample = t_sample
        self.disturbance = w

        self.optimizer = tf.keras.optimizers.RMSprop()
        self.u_ub = u_ub
        self.u_lb = u_lb
        self.input_dim = len(self.u_ub)

        #constraints:
        self.Gx = Gx
        self.gx = gx
        self.Gu = Gu
        self.gu = gu
        self.Gw = Gw
        self.gw = gw

        # Initialize state and control
        self.x0 = x0  # Initial state
        self.u0 = u0  # Initial control input


        self.distrubance_set = Polytope(Gw, gw)


        #due to using laguerre we change this form u to eta:
        self.u = tf.Variable(initial_value=np.zeros((self.H, self.input_dim)), name='u', trainable=True,
                             dtype=tf.float64)

        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)
        self.R = tf.convert_to_tensor(R, dtype=tf.float64)

        

        self.solving_times = {}

    def approximate_dynamics(self, model, x, u, t):
        
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        u = tf.convert_to_tensor(u, dtype=tf.float64)
        t = tf.convert_to_tensor(t, dtype=tf.float64)

        # Ensure tensors have the correct shape
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)  # Add batch dimension
        if len(u.shape) == 1:
            u = tf.expand_dims(u, axis=0)  # Add batch dimension
        if len(t.shape) == 0:
            t = tf.expand_dims(tf.expand_dims(t, axis=0), axis=1)

            
        inputs = tf.concat([t, x, u], axis=1)
        print(f"Concatenated input shape: {inputs.shape}")  # Debugging

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(u)
            x_pred = model(inputs)
            print(f"x_pred shape: {x_pred.shape}")  # Debugging

        A = tape.jacobian(x_pred, x)
        B = tape.jacobian(x_pred, u)
        print(f"A shape: {None if A is None else A.shape}")
        print(f"B shape: {None if B is None else B.shape}")

        if A is None or B is None:
            raise ValueError("Gradient computation failed. Check model output and input dependencies.")

        del tape
        return A, B

    def calculate_lqr_gain(self, A, B, Q, R):
   
        K, _, _ = control.lqr(A.numpy(), B.numpy(), Q.numpy(), R.numpy())
        return tf.convert_to_tensor(K, dtype=tf.float64)


    def disturbance_invariant_set(self, A_cl, alpha):

        Z = self.disturbance_set
        while True:
            # Check subset condition
            if all((A_cl @ Z.A) <= alpha * Z.b):
                break
            Z = minkowski_sum(A_cl @ Z, self.disturbance_set)
        return (1 / (1 - alpha)) * Z
    
    def compute_vertices(self, G, g):

        # Get dimension of the polytope
        n_dim = G.shape[1]

        # Enumerate all possible combinations of constraints
        vertices = []
        for indices in combinations(range(len(G)), n_dim):
            # Solve for the intersection of n_dim constraints
            A = G[list(indices), :]
            b = g[list(indices)]
            try:
                vertex = np.linalg.solve(A, b)
                # Check if the vertex satisfies all inequalities
                if np.all(G @ vertex <= g + 1e-8):  # Tolerance for numerical stability
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                # Skip singular matrices
                pass

        return np.array(vertices)

    def minkowski_difference(self, P1, P2):
    
        # Step 1: Compute vertices of both polytopes
        vertices_P1 = self.compute_vertices(P1.A, P1.b)
        vertices_P2 = self.compute_vertices(P2.A, P2.b)

        # Step 2: Compute the Minkowski difference vertices
        difference_vertices = []
        for v1 in vertices_P1:
            for v2 in vertices_P2:
                difference_vertices.append(v1 - v2)
        difference_vertices = np.unique(np.array(difference_vertices), axis=0)  # Remove duplicates

        # Step 3: Reconstruct polytope from vertices (use Convex Hull-like approximation)
        G, g = self.compute_halfspace_representation(difference_vertices)
        return Polytope(G, g)
    
    def compute_halfspace_representation(self, vertices):
        # Add a half-space representation using ConvexHull
        hull = ConvexHull(vertices)
        return hull.equations[:, :-1], -hull.equations[:, -1]
    
    def tighten_constraints(self, Gx, gx, Gu, gu, K, Z):
        # Create original state and input constraint polytopes
        X = Polytope(Gx, gx)
        U = Polytope(Gu, gu)

        # Tighten state constraints (X_bar = X - Z)
        X_bar = self.minkowski_difference(X, Z)  # Exact Minkowski difference for state constraints

        # Tighten input constraints (U_bar = U - KZ)
        KZ_vertices = np.dot(K, self.compute_vertices(Z.A, Z.b).T).T
        KZ_polytope = Polytope.from_vertices(KZ_vertices)
        U_bar = self.minkowski_difference(U, KZ_polytope)
        return X_bar, U_bar


    def compute_terminal_set(self, kmax=10):

        # Compute the terminal constraints
        GT = np.vstack([self.Gx, np.dot(self.Gu, self.K)])  # Combine state and input constraints
        gT = np.hstack([self.gx, self.gu])                 # Combine state and input bounds

        # Perform MPIS calculation
        terminal_set, _ = self.mpis(GT, gT, kmax)

        return terminal_set


    def mpis(self, GT, gT):
        kmax = 10
        # Initialize the sequence of terminal sets
        X = [Polytope(GT, gT)]  # Start with the initial terminal polytope

        for k in range(1, kmax + 1):
            # Compute the backward reachable set (BR set)
            current_terminal = {"GT": X[k - 1].A, "gT": X[k - 1].b}
            X_k = self.br_set(current_terminal)

            # Append the new polytope to the list
            X.append(X_k)

            # Check for convergence
            if X[k - 1] == X_k:
                break

        # Return the final terminal set and the number of iterations
        return X[-1], len(X)
    
    def br_set(self, terminal):

        # Compute the augmented constraints for the backward reachable set
        L = np.vstack([self.Gw, np.dot(terminal["GT"], self.A_cl)])  # Combine disturbance and state constraints
        r = np.hstack([self.gw, terminal["gT"] - np.dot(terminal["GT"], self.b_cl)])  # Combine bounds

        # Return the new polytope
        return Polytope(L, r)



    def costs(self, x_ref, x_pred):
        penalty_factor = 0.01
        #delta_u = self.u[1:] - self.u[:-1]

        J = tf.reduce_sum(tf.matmul(tf.square(x_ref - x_pred), self.Q)) \
            + tf.reduce_sum(tf.matmul(tf.square(self.u), self.R)) 
        return J

    def solve_ocp(self, x0, x_ref, iterations=1000, tol=1e-8):
        J_prev = -1
        for epoch in range(iterations):
            J, x_pred = self.optimization_step(x0, x_ref)
            if np.abs(J - J_prev) < tol:
                return J, x_pred
            J_prev = J

        return J, x_pred

    @tf.function
    def optimization_step(self, x0, x_ref):
        
        with tf.GradientTape() as tape:
            x_pred = self.sim_open_loop(x0, self.u, t_sample=self.t_sample, H=self.H)
            cost = self.costs(x_ref, x_pred)
        gradients = tape.gradient(cost, self.u)
        self.optimizer.apply_gradients(zip([gradients], [self.u]))
        self.ensure_constraints()

        return cost, x_pred
    
    def calculate_control(self, k, x, xb, x_ref, K):
        # Nominal control from optimization (u_nominal)
        u_nominal = self.u.numpy()[k] if k < self.H else -K @ (xb - x_ref)

        # Add robust correction
        u = u_nominal - K @ (x - xb)
        return u


    def feasibility_check(self, x0, X_tightened, U_tightened, Z):
        # Check if x0 satisfies the tightened state constraints
        if not np.all(X_tightened.A @ x0 <= X_tightened.b):
            raise ValueError(f"Initial state x0 does not satisfy the tightened state constraints: {X_tightened}")
        # Check if x0 is within the disturbance invariant set Z
        if not np.all(Z.A @ x0 <= Z.b):
            raise ValueError(f"Initial state x0 does not lie within the disturbance invariant set Z: {Z}")
        # If no issues, feasibility is satisfied
        if not np.all(U_tightened.A @ self.u0 <= U_tightened.b):
            raise ValueError(f"Initial control u0 does not satisfy the tightened input constraints: {U_tightened}")
        logging.info("Initial state x0 is feasible under the tightened constraints and disturbance invariant set.")



    def sim(self, x0, X_ref, T_ref):
        N = len(T_ref)
        

        X_mpc = np.zeros((N, len(x0)))
        X_pred = np.zeros((N, len(x0)))
        U_mpc = np.zeros((N, self.u.shape[1]))

        X_mpc[0] = x0
        X_pred[0] = x0
        U_mpc[0] = self.u.numpy()[0]


        for i, t in enumerate(T_ref[:-1]):
            start_time = time.time()
            # Approximate system dynamics (A and B matrices)
            A, B = self.approximate_dynamics(self.model, X_mpc[i], self.u[0], t)

            # Calculate LQR gain (K)
            self.K = self.calculate_lqr_gain(A, B, self.Q, self.R)

            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.H + 1])
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            # Compute the closed-loop system matrix for use in terminal set, etc.
            self.A_cl = self.A - tf.matmul(self.B, self.K)

            alph = 0.9
            Z = self.disturbance_invariant_set(self.A_cl, alph)
            
                
            # Tightened constraints
            X_tightened, U_tightened = self.tighten_constraints(self.Gx, self.gx, self.Gu, self.gu, self.K, Z)

            # Compute terminal set
            terminal_set = self.compute_terminal_set()
            G_terminal = terminal_set.A
            g_terminal = terminal_set.b

            # Feasibility check
            self.feasibility_check(x0, X_tightened, U_tightened, Z)


            # Compute control
            u_k = self.calculate_control(i, X_mpc[i], X_pred[0], X_ref[i], self.K)
            
            

            x_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample, self.disturbance)

            X_pred[i + 1] = x_pred[1]
            X_mpc[i + 1] = x_true
            U_mpc[i + 1] = u_k.numpy()

            log_str = f'\tIter: {str(i + 1).zfill(len(str(N - 1)))}/{N - 1},\tJ: {J:.2e},' \
                      f'\tt: {t + self.t_sample:.2f} s,'

            for i in range(len(u_k)):
                log_str = log_str + f'\tu{i + 1}: {u_k.numpy()[i]:.2f},'

            for i in range(int(len(x_true) / 2)):
                log_str = log_str + f'\tx{i + 1}(t, u): {x_true[i]:.2f},'

            log_str = log_str + f'\tOCP-solving-time: {ocp_solving_time:.2e} s'

            logging.info(log_str)

        return X_mpc, U_mpc, X_pred

    def sim_plant_system(self, x0, u, tau, disturbance):
        ivp_solution = solve_ivp(self.plant, [0, tau], x0, args=[u])
        z_true = np.moveaxis(ivp_solution.y[:, -1], -1, 0)
        w = np.random.normal(0, disturbance, size=z_true.shape)
        z_true_w = z_true + w
        return z_true_w

    def sim_open_loop(self, x0, u_array, t_sample, H):
        t = tf.constant(t_sample, dtype=tf.float64, shape=(1, 1))
        x_i = tf.expand_dims(x0, 0)
        X_pred = x_i

        for i in range(H):
            x = tf.concat((t, x_i, u_array[i:i + 1]), 1)
            x_pred = self.model(x)
            X_pred = tf.concat((X_pred, x_pred), 0)
            x_i = x_pred

        return X_pred

    def sim_open_loop_plant(self, x0, u_array, t_sample, H):
        x_i = x0
        X = x_i

        for i in range(H):
            x = self.sim_plant_system(x_i, u_array[i], t_sample, self.disturbance)
            X = np.vstack((X, x))
            x_i = x

        return X
