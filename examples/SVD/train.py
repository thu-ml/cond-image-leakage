import os

from torch.utils.tensorboard import SummaryWriter
import random
import logging
import inspect
import argparse
import datetime
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.training_utils import EMAModel
from svd.data.dataset import WebVid10M
from svd.training.loss import EDMLossDistributionLatent
from svd.models.utils import pixel2latent, encode_image
from svd.schedulers.edm import add_noise
from svd.training.utils import load_checkpoint, init_dist, set_seed, save_videos_grid
from svd.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

def main(
        debug: False,
        use_ema: False,
        output_dir: str,
        pretrained_model_path: str,
        train_data: Dict,
        motion_bucket_id: float=20.0,
        cfg_random_null_ratio: float = 0.1,
        resume_path: str = "",
        ema_decay: float = 0.9999,
        noise_scheduler_kwargs=None,
        max_train_steps: int = 100,
        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        train_batch_size: int = 1,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        checkpointing_steps: int = -1,
        mixed_precision_training: bool = True,
        global_seed: int = 42,
        max_grad_norm=1.0,
        beta_m=15,
        a=5
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    # Initialize distributed training
    local_rank = init_dist(launcher="pytorch", backend='nccl')
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    # Logging folder
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join(output_dir, folder_name)

    seed = global_seed
    set_seed(seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        writer = SummaryWriter(f"{output_dir}/logs")
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_path,subfolder="unet")

    # Move models to GPU
    vae.to(local_rank)
    image_encoder.to(local_rank)
    unet.to(local_rank)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # ema
    if use_ema:
        ema_unet = EMAModel(unet, decay=ema_decay)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # resume training
    if resume_path != "":
        if use_ema:
            load_checkpoint(unet, optimizer, lr_scheduler, ema_unet, filename=resume_path)
        else:
            load_checkpoint(unet, optimizer, lr_scheduler, filename=resume_path)
    else:
        global_step = 0
        first_epoch = 0

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    # Get the training dataset
    train_dataset = WebVid10M(**train_data)
    if not debug:
        distributed_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_processes,
            shuffle=True,
            seed=global_seed
        )

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=distributed_sampler,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )
    loss_fn = EDMLossDistributionLatent(P_mean=noise_scheduler_kwargs.P_mean, P_std=noise_scheduler_kwargs.P_std, \
                                            sigma_data=noise_scheduler_kwargs.sigma_data,beta_m=beta_m,a=a)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    unet.train()

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for step, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(local_rank)

        latents = pixel2latent(pixel_values, vae)

        # noise argumentation
        noisy_condition, noise_aug_strength = add_noise(pixel_values[:, 0, :, :, :].unsqueeze(dim=1), P_mean=noise_scheduler_kwargs.condition_P_mean, P_std=noise_scheduler_kwargs.condition_P_std)
        noisy_condition_latents = pixel2latent(noisy_condition, vae)/vae.config.scaling_factor

        # CLIP image embeddings
        encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].float(), feature_extractor, image_encoder)
        # Predict the noise residual and compute loss
        # Mixed-precision training
        loss = loss_fn(unet, latents, encoder_hidden_states, mixed_precision_training,pixel_values,vae,cfg_random_null_ratio,motion_bucket_id=motion_bucket_id,fps=train_data.fps)
        optimizer.zero_grad()

        # Backpropagate
        if mixed_precision_training:
            scaler.scale(loss).backward()
            """ >>> gradient clipping >>> """
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
            """ <<< gradient clipping <<< """
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            """ >>> gradient clipping >>> """
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
            """ <<< gradient clipping <<< """
            optimizer.step()

        progress_bar.update(1)
        global_step += 1

        # logging
        if use_ema:
            parameters = unet.parameters()
            ema_unet.step(parameters)

        if is_main_process:
            writer.add_scalar('Training loss', loss.item(), global_step=global_step)

        # Save checkpoint
        if global_step % checkpointing_steps == 0:
            now = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
            print(f'step {global_step}, time: {now}')
            save_path = os.path.join(output_dir, f"checkpoints")
            model_state_dict = unet.state_dict()
            state_dict = {
                "global_step": global_step,
                "state_dict": model_state_dict,
                'optimizer': optimizer.state_dict()
            }
            if use_ema:
                state_dict["ema_state"] = ema_unet.state_dict()
            torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
            logging.info(f"Saved state to {save_path} (global_step: {global_step})")

        logs = {"step_loss": loss.detach().item()}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(**config)
