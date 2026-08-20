import torch
import torch.nn as nn

class MultiObjectivePINNLoss(nn.Module):
    def __init__(self, w_v=0.005, w_phys=1.0, w_mono=0.5, w_smooth=0.1, w_soh=1.0):
        super().__init__()
        self.w_v = w_v
        self.w_phys = w_phys
        self.w_mono = w_mono
        self.w_smooth = w_smooth
        self.w_soh = w_soh
        self.mse = nn.MSELoss()

    def forward(self, outputs, v_measured, soh_target, r_est_measured=None):
        r_ohmic = outputs["r_ohmic"]
        eta_act = outputs["eta_act"]
        degradation = outputs["degradation"]
        soh_pred = outputs["soh_pred"]
        v_reconstructed = outputs["v_reconstructed"]
        
        # 1. Voltage Reconstruction Loss (Scaled)
        l_v = self.mse(v_reconstructed, v_measured)
        
        # 2. Physics Consistency Loss
        if r_est_measured is not None:
            valid_r = ~torch.isnan(r_est_measured)
            if valid_r.sum() > 0:
                l_phys = self.mse(r_ohmic[valid_r], r_est_measured[valid_r])
            else:
                l_phys = torch.tensor(0.0, device=v_measured.device)
        else:
            l_phys = torch.tensor(0.0, device=v_measured.device)
            
        # 3. Degradation Monotonicity Loss (dD/dt >= 0)
        dD = degradation[1:] - degradation[:-1]
        l_mono = torch.mean(torch.relu(-dD))
        
        # 4. Temporal Smoothness Loss
        l_smooth = torch.mean(dD**2)
        
        # 5. SoH Supervision Loss
        l_soh = self.mse(soh_pred, soh_target)
        
        # Total Balanced Loss
        total_loss = (
            self.w_v * l_v + 
            self.w_phys * l_phys + 
            self.w_mono * l_mono + 
            self.w_smooth * l_smooth + 
            self.w_soh * l_soh
        )
        
        return total_loss, {
            "l_voltage": l_v.item(),
            "l_physics": l_phys.item(),
            "l_mono": l_mono.item(),
            "l_smooth": l_smooth.item(),
            "l_soh": l_soh.item(),
            "total_loss": total_loss.item()
        }
