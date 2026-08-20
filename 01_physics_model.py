import numpy as np
import torch

class PEMPhysicsModel:
    def __init__(self, n_cells=360, e0=1.229):
        self.n_cells = n_cells
        self.e0 = e0

    def compute_e_nernst_numpy(self, T_celsius, P_kPa):
        T_c = np.nan_to_num(T_celsius, nan=58.0)
        P_k = np.nan_to_num(P_kPa, nan=96.0)
        T_k = np.clip(T_c + 273.15, 263.15, 393.15)
        P_k = np.clip(P_k, 80.0, 120.0)
        
        e_cell = self.e0 - 0.00085 * (T_k - 298.15) + (4.31e-5 * T_k * np.log(P_k / 101.325))
        return self.n_cells * e_cell

    def compute_e_nernst_torch(self, T_celsius, P_kPa):
        T_c = torch.nan_to_num(T_celsius, nan=58.0)
        P_k = torch.nan_to_num(P_kPa, nan=96.0)
        T_k = torch.clamp(T_c + 273.15, min=263.15, max=393.15)
        P_k = torch.clamp(P_k, min=80.0, max=120.0)
        
        e_cell = self.e0 - 0.00085 * (T_k - 298.15) + (4.31e-5 * T_k * torch.log(P_k / 101.325))
        return self.n_cells * e_cell

    def reconstruct_v_stack_torch(self, I_stack, T_celsius, P_kPa, R_ohmic, eta_act):
        E_nernst = self.compute_e_nernst_torch(T_celsius, P_kPa)
        I_s = torch.nan_to_num(I_stack, nan=0.0)
        r_o = torch.nan_to_num(R_ohmic, nan=0.10)
        eta_a = torch.nan_to_num(eta_act, nan=5.0)
        
        V_ohmic = I_s * r_o
        V_hat = E_nernst - V_ohmic - eta_a
        return V_hat
