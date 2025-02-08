import logging
import time
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

class MPC:
    """
    Class used to represent a Model Predictive Controller.
    """

    def __init__(self, plant, model, u_ub, u_lb,x_ub, x_lb, t_sample=0.1, H=10, Nu = 5, alpha = 0.9, N_lag = 3, w = 0.01,
                 Q=tf.eye(1, dtype=tf.float64), R=tf.eye(1, dtype=tf.float64)):
        self.plant = plant
        self.model = model
        self.Np = H
        self.Nu = Nu
        self.t_sample = t_sample

        self.optimizer = tf.keras.optimizers.RMSprop()
        self.u_ub = u_ub
        self.u_lb = u_lb
        self.x_ub = x_ub
        self.x_lb = x_lb
        self.disturbance = w
        self.input_dim = len(self.u_ub)

        self.alpha = alpha
        self.N_lag = N_lag

        self.L = tf.convert_to_tensor(self.compute_lagbase(N_lag, alpha, H), dtype=tf.float64)

        self.eta = tf.Variable(initial_value=np.zeros((N_lag, self.input_dim)), name='eta', trainable=True,
                             dtype=tf.float64)

        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)
        self.R = tf.convert_to_tensor(R, dtype=tf.float64)

        self.optimizer = tf.keras.optimizers.RMSprop()
        self.solving_times = {}


    def compute_lagbase(self, N_lag, alpha, Nu):
        """
        Compute the Laguerre basis matrix.

        Args:
            N_lag (int): Number of Laguerre functions (basis size).
            alpha (float): Laguerre parameter (0 < alpha < 1 for stability).
            H (int): Prediction horizon.

        Returns:
            np.ndarray: Laguerre basis matrix of shape (H, N_lag).
        """
        beta = 1 - alpha**2
        L0 = beta ** 0.5 * np.array([(-alpha) ** i for i in range(N_lag)])
        L0 = L0.reshape(-1, 1)
        A = np.diag([alpha] * N_lag)
        lower_diag = np.zeros((N_lag, N_lag))
        for row in range(1, N_lag):  
            for col in range(row):  
                lower_diag[row, col] = (-alpha ** (row - col - 1)) * beta
        A_lag = A + lower_diag

        L = L0
        current_L = L0
        for _ in range(1, Nu):
            current_L = A_lag @ current_L
            L = np.hstack((L, current_L))
        return L.T  # Shape (Nu, N_lag)

    def expand_u(self, eta):
        """
        Compute the control inputs u over the prediction horizon from eta and Laguerre basis.

        Args:
            eta (tf.Variable): Optimization variable (N_lag x input_dim).

        Returns:
            tf.Tensor: Control inputs (Np x input_dim).
        """
        return tf.matmul(self.L, eta)  # Shape (Np, input_dim)


    def costs(self, x_ref, x_pred,u):
        """
        Represents the MPC cost function, which is composed of the step cost and the final cost.

        :param x_ref: reference states
        :param x_pred: predicted states
        :return: J: cost value
        """
        state_violation = tf.reduce_sum(tf.nn.relu(x_pred - self.x_ub)) + tf.reduce_sum(tf.nn.relu(self.x_lb - x_pred))
        
        J = tf.reduce_sum(tf.square(x_ref - x_pred) @ self.Q) \
            + tf.reduce_sum(tf.square(u) @ self.R)\
            + (0.01 * state_violation)

        return J

    def solve_ocp(self, x0, x_ref, iterations=1000, tol=1e-8):
        J_prev = -1
        for epoch in range(iterations):
            J, x_pred = self.optimization_step(x0, x_ref)
            self.ensure_constraints(self.expand_u(self.eta))
            if np.abs(J - J_prev) < tol:
                break
            J_prev = J

        return J, x_pred

    @tf.function
    def optimization_step(self, x0, x_ref):
        with tf.GradientTape() as tape:
            u = self.expand_u(self.eta)  # Shape (Np, input_dim)
            x_pred = self.sim_open_loop(x0, u, self.t_sample, self.Np)
            cost = self.costs(x_ref, x_pred,u)
        gradients = tape.gradient(cost, self.eta)
        self.optimizer.apply_gradients(zip([gradients], [self.eta]))
        constrained_u = self.ensure_constraints(u)
        return cost, x_pred

    def ensure_constraints(self,u):
        lower_violation = tf.where(u < self.u_lb, self.u_lb - u, 0.0)  # Positive where u < u_lb
        upper_violation = tf.where(u > self.u_ub, u - self.u_ub, 0.0)  # Positive where u > u_ub

        # Adjust eta dynamically if any violations are detected
        if tf.reduce_any(lower_violation > 0) or tf.reduce_any(upper_violation > 0):
            # Reduce eta globally (or adjust per element as needed)
            self.eta.assign(self.eta * (1 - tf.reduce_mean(tf.abs(lower_violation + upper_violation))))

        # Apply constraints: clamp u to [u_lb, u_ub]
        constrained_u = tf.where(u < self.u_lb, self.u_lb, u)  # Clamp to lower bound
        constrained_u = tf.where(constrained_u > self.u_ub, self.u_ub, constrained_u)  # Clamp to upper bound

        return constrained_u

    def sim(self, x0, X_ref, T_ref):
        N = len(T_ref)
        state_dim = len(x0)

        X_mpc = np.zeros((N, state_dim))
        X_pred = np.zeros((N, state_dim))
        U_mpc = np.zeros((N, self.input_dim))

        X_mpc[0] = x0
        X_pred[0] = x0
        U_mpc[0] = self.expand_u(self.eta)[0].numpy()

        for i, t in enumerate(T_ref[:-1]):
            start_time = time.time()

            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.Np + 1])
            u = self.expand_u(self.eta)
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            u_k = u[0]

            x_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample, self.disturbance)

            X_pred[i + 1] = x_pred[1].numpy()
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
