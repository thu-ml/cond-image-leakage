import torch
from ..schedulers.edm import add_noise
from ..models.utils import VScalingWithEDMcNoise
from ..models.utils import pixel2latent
import random

# sample from a logit-normal distribution
def logit_normal_sampler(m, s=1, beta_m=15, sample_num=1000000):
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples

# the $\mu(t)$ function
def mu_t(t, a=5, mu_max=1):
    t = t.to('cpu')
    return 2 * mu_max * t**a - mu_max
    
# get $\beta_s$ for TimeNoise
def get_beta_s(t, a,beta_m):
    mu = mu_t(t,a=a)
    sigma_s = logit_normal_sampler(m=mu, sample_num=t.shape[0], beta_m=beta_m)
    return sigma_s

# loss function for TimeNoise of the paper: https://arxiv.org/pdf/2406.15735.  
class EDMLossTimeNoise:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,beta_m=15,a=5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.beta_m=beta_m
        self.a=a
        
    def __call__(self, unet, latents, encoder_hidden_states,
                 mixed_precision_training,pixel_values,vae,cfg_random_null_ratio,
                 motion_bucket_id,fps):
        # add noise to ground-truth video
        noisy_latents, sigma = add_noise(latents, self.P_mean, self.P_std)
        c_skip, c_out, c_in, c_noise = VScalingWithEDMcNoise(sigma)
        scaled_inputs = noisy_latents * c_in
        
        # conditional image
        img_condition = pixel_values[:, 0, :, :, :].unsqueeze(dim=1)
        img_condition_latents = pixel2latent(img_condition, vae)/vae.config.scaling_factor
        
        # applying TimeNoise. Add logit-normal noise to the conditional image.
        noise_aug_strength =get_beta_s(sigma/700,self.a,self.beta_m).reshape([latents.shape[0], 1, 1, 1, 1]).to(latents.device)
        rnd_normal = torch.randn([img_condition_latents.shape[0], 1, 1, 1, 1], device=img_condition_latents.device)
        noisy_condition_latents =img_condition_latents + noise_aug_strength * rnd_normal
        
        # classifier-free guidance
        if cfg_random_null_ratio > 0.0:
            p = random.random()
            noisy_condition_latents = noisy_condition_latents if p > cfg_random_null_ratio else torch.zeros_like(noisy_condition_latents)
            encoder_hidden_states = encoder_hidden_states if p > cfg_random_null_ratio else torch.zeros_like(encoder_hidden_states)

        # Repeat the condition latents for each frame so we can concatenate them with the noise
        noisy_condition_latents = noisy_condition_latents.repeat(1, latents.shape[1], 1, 1, 1)
        

        batch_size = noise_aug_strength.shape[0]
        motion_score = torch.tensor([motion_bucket_id]).repeat(batch_size).to(latents.device)
        fps = torch.tensor([fps]).repeat(batch_size).to(latents.device)
        added_time_ids = torch.stack([fps, motion_score, noise_aug_strength.reshape(batch_size)], dim=1)
    
        scaled_inputs = torch.cat([scaled_inputs,noisy_condition_latents], dim=2)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # calculate loss
        with torch.cuda.amp.autocast(enabled=mixed_precision_training):
            model_pred = unet(
                scaled_inputs, c_noise, encoder_hidden_states, added_time_ids=added_time_ids)["sample"]

            pred = model_pred * c_out + c_skip * noisy_latents
            loss = torch.mean((weight.float() * (pred.float() - latents.float()) ** 2))

        return loss
    
    

# baseline loss function 
class EDMLossBaseline:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
    def __call__(self, unet, latents, encoder_hidden_states,
                 mixed_precision_training,pixel_values,vae,cfg_random_null_ratio,
                 motion_bucket_id,fps):
        # add noise to ground-truth video
        noisy_latents, sigma = add_noise(latents, self.P_mean, self.P_std)
        c_skip, c_out, c_in, c_noise = VScalingWithEDMcNoise(sigma)
        scaled_inputs = noisy_latents * c_in
        
        # baseline noise argumentation
        noisy_condition, noise_aug_strength = add_noise(pixel_values[:, 0, :, :, :].unsqueeze(dim=1), P_mean=self.P_mean, P_std=self.P_std)
        noisy_condition_latents = pixel2latent(noisy_condition, vae)/vae.config.scaling_factor

        # classifier-free guidance
        if cfg_random_null_ratio > 0.0:
            p = random.random()
            noisy_condition_latents = noisy_condition_latents if p > cfg_random_null_ratio else torch.zeros_like(noisy_condition_latents)
            encoder_hidden_states = encoder_hidden_states if p > cfg_random_null_ratio else torch.zeros_like(encoder_hidden_states)

        # Repeat the condition latents for each frame so we can concatenate them with the noise
        noisy_condition_latents = noisy_condition_latents.repeat(1, latents.shape[1], 1, 1, 1)
        
        batch_size = noise_aug_strength.shape[0]
        motion_score = torch.tensor([motion_bucket_id]).repeat(batch_size).to(latents.device)
        fps = torch.tensor([fps]).repeat(batch_size).to(latents.device)
        added_time_ids = torch.stack([fps, motion_score, noise_aug_strength.reshape(batch_size)], dim=1)
    
        scaled_inputs = torch.cat([scaled_inputs,noisy_condition_latents], dim=2)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # calculate loss
        with torch.cuda.amp.autocast(enabled=mixed_precision_training):
            model_pred = unet(
                scaled_inputs, c_noise, encoder_hidden_states, added_time_ids=added_time_ids)["sample"]

            pred = model_pred * c_out + c_skip * noisy_latents
            loss = torch.mean((weight.float() * (pred.float() - latents.float()) ** 2))

        return loss
 