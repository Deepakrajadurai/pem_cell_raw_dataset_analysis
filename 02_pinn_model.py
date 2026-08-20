import torch
import torch.nn as nn

class PEM_PINN_Encoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU()
        )
        
        # Degradation state head D in [0.0, 0.35]
        self.degradation_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Activation Overpotential head eta_act in [1.0, 40.0] V
        self.eta_act_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.net(x)
        
        # 1. Structural Degradation State D(t)
        degradation = 0.35 * self.degradation_head(feat).squeeze(-1)
        
        # 2. Ohmic Resistance is explicitly coupled to structural degradation D(t)
        # R_ohmic(t) = R_BOL * (1 + 2.5 * D(t))
        r_ohmic = 0.080 * (1.0 + 2.5 * degradation)
        
        # 3. Activation overpotential
        eta_act = 1.0 + 39.0 * self.eta_act_head(feat).squeeze(-1)
        
        # 4. SoH Target = 100 * (1 - D)
        soh_pred = (1.0 - degradation) * 100.0
        
        return {
            "r_ohmic": r_ohmic,
            "eta_act": eta_act,
            "degradation": degradation,
            "soh_pred": soh_pred
        }
