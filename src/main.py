import logging
import os

import numpy as np
import tensorflow as tf

from controller.mpc import MPC
from train_pinn import ManipulatorInformedNN
from utils.data import load_ref_trajectory, load_data
from utils.plotting import plot_input_sequence, plot_states, plot_absolute_error, animate
from utils.system import f

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info("TensorFlow version: {}".format(tf.version.VERSION))
    logging.info("Eager execution: {}".format(tf.executing_eagerly()))

    # Paths
    resources_path = os.path.join('../resources')
    data_path = os.path.join(resources_path, 'data.npz')
    weights_path = os.path.join('../resources/weights')

    lb, ub, input_dim, output_dim, _, _, _, _ = load_data(data_path)

    # Hyper parameter
    N_l = 4
    N_n = 64
    layers = [input_dim, *N_l * [N_n], output_dim]

    logging.info('MPC parameters:')
    H = 20 
    Nu = 10
    w = 0.01
    logging.info(f'\tH:\t{H}')
    u_ub = np.array([0.5, 0.5])
    u_lb = - u_ub

    x_ub = np.array([3.14, 3.14, 20, 20])
    x_lb = - x_ub

    X_ref, T_ref = load_ref_trajectory(resources_path)
    T_ref = T_ref

    x0 = X_ref[0]
    u0 = np.array([0.0 , 0.0]) 
    T_ref = T_ref[:-H, 0]

    #constraints:
    Gx = np.array([[1, 0,0,0], [-1, 0,0,0], [0, 1,0,0], [0, -1,0,0],[0, 0,0,0], [0, 0,0,0], [0, 0,0,0], [0, 0,0,0]])
    gx =  np.array([3.14, 3.14, 3.14, 3.14, 0, 0, 0, 0])

    Gu = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    gu = np.array([0.5, 0.5, 0.5, 0.5])

    Gw = np.array([[1, 0,0,0], [-1, 0,0,0], [0, 1,0,0], [0, -1,0,0],[0, 0,1,0], [0, 0,-1,0], [0, 0,0,1], [0, 0,0,-1]])
    gw = np.array([w,w,w,w,w,w,w,w])



    #laguerre:
    alpha = 0.9
    N_lag = 10

    tau = T_ref[1] - T_ref[0]
    logging.info(f'\ttau:\t{tau}')

    # Initialization
    pinn = ManipulatorInformedNN(layers, lb, ub)
    # Load pretrained weights
    pinn.load_weights(weights_path)

#    controller = MPC(f, pinn.model, u_ub=u_ub, u_lb=u_lb, x_ub=x_ub, x_lb=x_lb,
#                     t_sample=tau, H=H, Nu = Nu, w=w, 
#                     Q=tf.linalg.tensor_diag(tf.constant([1, 1, 0, 0], dtype=tf.float64)),
#                     R=1e-6 * tf.eye(2, dtype=tf.float64))

    controller = MPC(f, pinn.model, u_ub=u_ub, u_lb=u_lb, x_ub=x_ub, x_lb=x_lb,
                     t_sample=tau, H=H, Nu = Nu, alpha = 0.9, N_lag = 3, w=w, 
                     Q=tf.linalg.tensor_diag(tf.constant([1, 1, 0, 0], dtype=tf.float64)),
                     R=1e-6 * tf.eye(2, dtype=tf.float64))


#
#    controller = MPC2(f, pinn.model, x0 ,u0, u_ub=u_ub, u_lb=u_lb, Gx = Gx, gx =gx, Gu = Gu, gu = gu, Gw =Gw, gw = gw, 
#                     t_sample=tau, H=H, w=w, 
#                     Q=tf.linalg.tensor_diag(tf.constant([1, 1, 0, 0], dtype=tf.float64)),
#                     R=1e-6 * tf.eye(2, dtype=tf.float64))
   
    # Testing self loop prediction
    H_sl = 20

    # Generate self loop prediction input sequence
    U1_sl = 0.5 * np.sin(np.linspace(0, 2 * np.pi, H_sl))
    U2_sl = - U1_sl
    U_sl = np.hstack((U1_sl[:, np.newaxis], U2_sl[:, np.newaxis]))

    # Initial state
    x0_sl = np.zeros(4)

    # Simulate plant system
    X_ref_sl = controller.sim_open_loop_plant(x0_sl, U_sl,
                                              t_sample=tau,
                                              H=H_sl)

    # Simulate PINN system
    X_sl = controller.sim_open_loop(x0_sl, U_sl,
                                    t_sample=tau,
                                    H=H_sl)

    T_sl = np.arange(0., H_sl * tau + tau, tau)
    plot_input_sequence(T_sl, U=np.vstack((U_sl, U_sl[-1:, :])))
    plot_states(T_sl, X_ref_sl, X_sl)
    plot_absolute_error(T_sl, X_ref_sl, X_sl)

    # Testing closed loop
    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref)

    plot_input_sequence(T_ref, U_mpc)
    plot_states(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    plot_absolute_error(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    animate(X_ref[:-H], [X_mpc], ['MPC'], fps=1 / tau)
