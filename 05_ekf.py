import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, q_cov=1e-6, r_obs=400.0):
        # q_cov: Process noise covariance
        # r_obs: Measurement noise covariance
        self.q_cov = q_cov
        self.r_obs = r_obs
        
        # Initial State Estimate [Degradation D, R_ohmic, eta_act]
        self.x = np.array([0.0, 0.080, 10.0])
        self.P = np.eye(3) * 0.001

    def reset(self, init_soh=100.0, init_r=0.080, init_eta=10.0):
        init_d = (100.0 - init_soh) / 100.0
        self.x = np.array([init_d, init_r, init_eta])
        self.P = np.eye(3) * 0.001

    def step(self, v_meas, v_pinn_pred, pinn_soh_pred, pinn_r_ohmic, pinn_eta_act):
        x_pred = np.array([ (100.0 - pinn_soh_pred) / 100.0, pinn_r_ohmic, pinn_eta_act ])
        P_pred = self.P + np.eye(3) * self.q_cov
        
        # Voltage Innovation residual e_v = V_meas - V_pinn_pred
        e_v = v_meas - v_pinn_pred
        
        # Jacobian H = d(V_hat)/dx
        H = np.array([ -0.5, -0.05, -0.01 ])
        
        S = H.dot(P_pred).dot(H.T) + self.r_obs
        K = P_pred.dot(H.T) / S
        
        # Update Step with Innovation Gating (clamp e_v to [-30V, +30V])
        e_v_clamped = np.clip(e_v, -30.0, 30.0)
        self.x = x_pred + K * e_v_clamped
        self.x[0] = np.clip(self.x[0], 0.0, 0.35)
        self.P = (np.eye(3) - np.outer(K, H)).dot(P_pred)
        
        corrected_soh = (1.0 - self.x[0]) * 100.0
        corrected_soh = np.clip(corrected_soh, 65.0, 100.0)
        
        return {
            "ekf_soh": corrected_soh,
            "ekf_r_ohmic": self.x[1],
            "ekf_eta_act": self.x[2],
            "innovation": e_v,
            "kalman_gain_mag": np.linalg.norm(K)
        }
