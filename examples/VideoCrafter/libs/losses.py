import random
import torch
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import numpy as np


def logit_normal_sampler(
    m, s=1, betam=0.9977, sample_num=1000000,
):
    y_samples = torch.randn(sample_num) * s + m
    x_samples = betam * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples


def mu_t(t, mu_t_type="linear", mu_max=1.5):
    t = t.to("cpu")
    if mu_t_type == "linear":
        return 2 * mu_max * t - mu_max
    elif mu_t_type == "Convex_10":
        return 2 * mu_max * t**10 - mu_max
    elif mu_t_type == "Convex_5":
        return 2 * mu_max * t**5 - mu_max
    elif mu_t_type == "sigmoid":
        return (
            2 * mu_max * np.exp(50 * (t - 0.8)) / (1 + np.exp(50 * (t - 0.8))) - mu_max
        )

def get_alpha_s_and_sigma_s(t, mut_type):
    mu = mu_t(t, mu_t_type=mut_type)
    sigma_s = logit_normal_sampler(m=mu, sample_num=t.shape[0], whether_paint=False)
    alpha_s = torch.sqrt(1 - sigma_s**2)
    return alpha_s, sigma_s


def dynamicrafter_loss(model_context, data_context, config, accelerator):

    batch = next(data_context["train_data_generator"])
    model = model_context["model"]
    local_rank = model_context["device"]
    dtype = model_context["dtype"]

    @torch.no_grad()
    @torch.autocast("cuda")
    def get_latent_z(model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, "b c t h w -> (b t) c h w")
        z = model.encode_first_stage(x)
        z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
        return z

    with torch.autocast("cuda", dtype=dtype):
        p = random.random()
        pixel_values = batch["pixel_values"].to(local_rank)
        # classifier-free guidance
        batch["text"] = [
            name if p > config.cfg_random_null_ratio else "" for name in batch["text"]
        ]
        prompts = batch["text"]
        fs = batch["fps"].to(local_rank)
        batch_size = pixel_values.shape[0]
        z = get_latent_z(model, pixel_values)  # b c t h w

        # add noise
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=z.device).long()
        noise = torch.randn_like(z)
        noisy_z = model.q_sample(z, t, noise=noise)

        # condition
        # classifier-free guidance
        img = (
            pixel_values[:, :, 0]
            if p > config.cfg_random_null_ratio
            else torch.zeros_like(pixel_values[:, :, 0])
        )
        img_emb = model.embedder(img)  ## blc
        img_emb = model.image_proj_model(img_emb)
        cond_emb = model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

        img_cat_cond = z[:, :, :1, :, :]
        # add noise on condition
        if config.condition_type is not None:
            alpha_s, sigma_s = get_alpha_s_and_sigma_s(t / 1000.0, config.mu_type)
            condition_noise = torch.randn_like(img_cat_cond)
            alpha_s = alpha_s.reshape([batch_size, 1, 1, 1, 1]).to(local_rank)
            sigma_s = sigma_s.reshape([batch_size, 1, 1, 1, 1]).to(local_rank)
            img_cat_cond = alpha_s * img_cat_cond + sigma_s * condition_noise

        # classifier-free guidance
        img_cat_cond = (
            img_cat_cond
            if p > config.cfg_random_null_ratio
            else torch.zeros_like(img_cat_cond)
        )
        img_cat_cond = repeat(
            img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2]
        )
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

        model_pred = model.apply_model(noisy_z, t, cond, fs=fs)
        loss = torch.mean(((model_pred.float() - noise.float()) ** 2))

    return {
        "loss": loss,
    }
