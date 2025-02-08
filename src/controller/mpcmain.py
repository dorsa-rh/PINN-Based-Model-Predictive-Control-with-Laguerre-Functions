import logging
import time
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp

class MPC:
    """
    Class used to represent a Model Predictive Controller.
    """

    def __init__(self, plant, model, u_ub, u_lb,x_ub, x_lb, t_sample=0.1, H=10, Nu = 5, w = 0.01,
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

        self.u = tf.Variable(initial_value=np.zeros((self.Nu, self.input_dim)), name='u', trainable=True,
                             dtype=tf.float64)

        self.Q = tf.convert_to_tensor(Q, dtype=tf.float64)
        self.R = tf.convert_to_tensor(R, dtype=tf.float64)

        self.solving_times = {}

    def expand_u(self, u):
        """
        Expand the control vector u (size Nu) to size Np by repeating the last control
        input for steps beyond Nu.
        """
        if self.Np <= self.Nu:
            return u[:self.Np]

        # Repeat the last control input for steps beyond Nu
        last_input = u[-1][tf.newaxis, :]  # Shape (1, input_dim)
        repeat_count = self.Np - self.Nu
        u_expanded = tf.concat([u, tf.tile(last_input, [repeat_count, 1])], axis=0)
        return u_expanded

    def costs(self, x_ref, x_pred):
        """
        Represents the MPC cost function, which is composed of the step cost and the final cost.

        :param x_ref: reference states
        :param x_pred: predicted states
        :return: J: cost value
        """
        state_violation = tf.reduce_sum(tf.nn.relu(x_pred - self.x_ub)) + tf.reduce_sum(tf.nn.relu(self.x_lb - x_pred))
        
        J = tf.reduce_sum(tf.square(x_ref - x_pred) @ self.Q) \
            + tf.reduce_sum(tf.square(self.u) @ self.R)\
            + (0.01 * state_violation)

        return J

    def solve_ocp(self, x0, x_ref, iterations=1000, tol=1e-8):
        J_prev = -1
        for epoch in range(iterations):
            J, x_pred = self.optimization_step(x0, x_ref)
            if np.abs(J - J_prev) < tol:
                break
            J_prev = J

        return J, x_pred

    @tf.function
    def optimization_step(self, x0, x_ref):
        with tf.GradientTape() as tape:
            u_expanded = self.expand_u(self.u)  # Size => (Np, input_dim)
            x_pred = self.sim_open_loop(x0, u_expanded, t_sample=self.t_sample, H=self.Np)
            cost = self.costs(x_ref, x_pred)
        gradients = tape.gradient(cost, self.u)
        self.optimizer.apply_gradients(zip([gradients], [self.u]))
        self.ensure_constraints()

        return cost, x_pred

    def ensure_constraints(self):
        for k in range(self.Nu):
            for i, u_ub_i in enumerate(self.u_ub):
                if self.u[k, i] > u_ub_i:
                    self.u[k, i].assign(u_ub_i)

            for i, u_lb_i in enumerate(self.u_lb):
                if self.u[k, i] < u_lb_i:
                    self.u[k, i].assign(u_lb_i)

    def sim(self, x0, X_ref, T_ref):
        N = len(T_ref)
        state_dim = len(x0)

        X_mpc = np.zeros((N, state_dim))
        X_pred = np.zeros((N, state_dim))
        U_mpc = np.zeros((N, self.input_dim))

        X_mpc[0] = x0
        X_pred[0] = x0
        U_mpc[0] = self.u[0].numpy()

        for i, t in enumerate(T_ref[:-1]):
            start_time = time.time()

            J, x_pred = self.solve_ocp(X_mpc[i], X_ref[i:i + self.Np + 1])
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            u_k = self.u[0]

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
