import torch

def add_noise(inputs, P_mean, P_std):#input [B,T,C,H,W]
    noise = torch.randn_like(inputs) #N(0,I)
    rnd_normal = torch.randn([inputs.shape[0], 1, 1, 1, 1], device=inputs.device) #N(0,I)采样 [B,1,1,1,1]
    sigma = (rnd_normal * P_std + P_mean).exp() # N(P_mean,P_std)
    noisy_inputs = inputs + noise * sigma
    return noisy_inputs, sigma