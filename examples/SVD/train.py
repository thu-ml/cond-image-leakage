import os
from torch.utils.tensorboard import SummaryWriter
import logging
import inspect
import argparse
import datetime
from einops import rearrange
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
from diffusers.pipelines import StableVideoDiffusionPipeline
from svd.data.dataset import WebVid10M,ImageDataset
from svd.training.loss import EDMLossTimeNoise,EDMLossBaseline
from svd.models.utils import pixel2latent, encode_image
from svd.training.utils import load_checkpoint, init_dist, set_seed,load_PIL_images,save_videos_grid
from svd.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

def main(
        use_ema: False,
        output_dir: str,
        pretrained_model_path: str,
        train_data: Dict,
        validation_folder: str,
        motion_bucket_id: float=20.0,
        cfg_random_null_ratio: float = 0.1,
        resume_path: str = "",
        ema_decay: float = 0.9999,
        noise_scheduler_kwargs=None,
        max_train_steps: int = 10000,
        validation_steps: int = 10000,
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
        max_grad_norm: float = 1.0,
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

    global_step = 0

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    train_dataset = WebVid10M(**train_data)
    
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
    # get the test dataset
    test_dataset = ImageDataset(validation_folder)
    test_distributed_sampler = DistributedSampler(
        test_dataset,
        shuffle=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=test_distributed_sampler,
        num_workers=num_processes,
        pin_memory=True,
        drop_last=False,
    )
    validation_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae).to(local_rank)
    
    # loss function for TimeNoise
    loss_fn = EDMLossTimeNoise(P_mean=noise_scheduler_kwargs.P_mean, P_std=noise_scheduler_kwargs.P_std, \
                                           sigma_data=noise_scheduler_kwargs.sigma_data,beta_m=noise_scheduler_kwargs.beta_m,a=noise_scheduler_kwargs.a)
    
    # baseline (train without TimeNoise): 
    # loss_fn = EDMLossBaseline(P_mean=noise_scheduler_kwargs.P_mean, P_std=noise_scheduler_kwargs.P_std,sigma_data=noise_scheduler_kwargs.sigma_data)
    
    # DDP wrapper
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    unet.train()

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for step, batch in enumerate(train_dataloader):
        pixel_values = batch["pixel_values"].to(local_rank)

        latents = pixel2latent(pixel_values, vae)
        
        # CLIP image embeddings
        encoder_hidden_states = encode_image(pixel_values[:, 0, :, :, :].float(), feature_extractor, image_encoder)
        # Predict the noise residual and compute loss
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

        # Periodically validation
        if global_step % validation_steps == 0:
            unet.eval()
            if use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())

            resolution = tuple(map(int, train_data.sample_size.split(',')))

            for _, test_batch in enumerate(test_dataloader):
                paths = test_batch['path']
                names = test_batch['name']
                images_list = load_PIL_images(paths, resolution) # resolution, need to check
                generator = torch.manual_seed(42)
                samples = validation_pipeline(images_list, output_type="pt", generator=generator, height=resolution[0],
                                width=resolution[1],  num_frames=train_data.sample_n_frames, motion_bucket_id=motion_bucket_id).frames
                samples = torch.stack(samples)

                for sample, name in zip(samples, names):
                    name = name.split('.')[0]
                    save_path = f"{output_dir}/samples/{global_step}_ema_{use_ema}/{name}.mp4"
                    sample = rearrange(sample, "t c h w -> c t h w").unsqueeze(dim=0)
                    save_videos_grid(sample.cpu(), save_path, n_rows=1)
                    print(f'the sample has been saved in {save_path}')

            if use_ema:
                ema_unet.restore(unet.parameters())

            unet.train()
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
