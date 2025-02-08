# PINN-Based Model Predictive Control with Laguerre Functions

This project presents an advanced approach to **Nonlinear Model Predictive Control (NMPC)** for multi-link manipulators by integrating **Physics-Informed Neural Networks (PINNs)** with **Laguerre functions**. Our method aims to improve computational efficiency and control performance in trajectory tracking tasks.

## Overview

Building upon the foundational work detailed in [this repository](https://github.com/Jonas-Nicodemus/PINNs-based-MPC), which introduces PINNs into NMPC for multi-body dynamics, our project introduces several key enhancements:

- **Laguerre Function Integration:**  
  We employ **Laguerre functions** to parameterize control inputs, reducing the dimensionality of the control problem and enhancing computational efficiency.

- **Control and Prediction Horizons:**  
  By adjusting the **control horizon (Nu)** and **prediction horizon (Np)**, we achieve a balance between performance and computational load, tailoring the controller's responsiveness to specific application needs.

- **State Constraints:**  
  Incorporation of **explicit state constraints** ensures the system operates within safe and feasible boundaries, enhancing the reliability of the control strategy.

- **Realistic Disturbance Implementation:**  
  To improve realism, we introduce **a disturbance to the states received from the plant**. This simulates real-world uncertainties, allowing the controller to adapt to variations in system dynamics.

---

## 🔍 Key Differences from the Referenced Project

While the referenced project lays the groundwork for using PINNs in NMPC, our approach distinguishes itself through:

1. **Laguerre Function Application:**  
   The original project does **not** utilize Laguerre functions; our integration of these functions streamlines the control input parameterization, leading to improved computational performance.

2. **Dynamic Adjustment of Horizons:**  
   We explore various configurations of **prediction (Np) and control (Nu) horizons**, analyzing their impact on system performance and computational demands, whereas the original project maintains fixed horizons.

3. **Enhanced Constraint Handling:**  
   Our controller explicitly incorporates **state constraints**, ensuring adherence to safety and operational limits, a feature not emphasized in the original implementation.

4. **Disturbance Modeling in Simulation:**  
   Unlike the referenced work, we introduce **stochastic noise (disturbance) in the plant model** to simulate real-world uncertainties. This is implemented by adding Gaussian noise to the plant state outputs.

---

## 📊 Results

Through extensive simulations, we demonstrate that our enhanced **NMPC framework** achieves superior **trajectory tracking performance** for multi-link manipulators.  
The integration of **Laguerre functions** and the **dynamic adjustment** of control parameters contribute to **reduced computational load** and **improved adherence to desired trajectories**.

### 🔹 Comparison of Different **Np** and **Nu** Configurations  
_This section highlights the impact of different prediction and control horizons on system performance._

![Np and Nu Comparison](path/to/Np_Nu_comparison_image.png)

### 🔹 Path Following Results  
_Illustrating the effectiveness of the proposed approach for various reference trajectories._

![Path Following](path/to/path_following_results.png)

### 🔹 Simulation with Disturbance  
_Showcasing how the controller adapts to system uncertainties due to introduced disturbances._

![Disturbance Simulation](path/to/disturbance_simulation.png)

### 🔹 Simulation GIF  
_Animated demonstration of the manipulator's response under the proposed control strategy._

![Simulation](path/to/simulation_gif.gif)

---

## 📂 Repository Structure

