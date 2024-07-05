import os
import torch
from svd.inference.pipline_CILsvd import StableVideoDiffusionCILPipeline
from svd.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from svd.training.utils import save_videos_grid, load_PIL_images
from einops import rearrange
from svd.data.dataset import ImageDataset
from torch.utils.data.distributed import DistributedSampler
from svd.training.utils import init_dist, set_seed
import torch.distributed as dist
from diffusers.training_utils import EMAModel
from svd.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
import argparse
from omegaconf import OmegaConf

def load_model(model, ema, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        if 'ema_state' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state'])

def sampling(
        data_dir,
        scheduler_path, 
        num_step,
        pretrained_model_path,
        checkpoint_path,
        save_dir, 
        resolution,
        batch_size,
        use_ema = True,
        num_processes=8, 
        local_rank=0, 
        motion_bucket_id=20,
        fps=3,
        num_frames=16,
        seed=42,
        analytic_path=''
    ):
    
    scheduler = EulerDiscreteScheduler.from_config(scheduler_path)
    
    if checkpoint_path is not None:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", torch_dtype=torch.float16)
        ema_unet = EMAModel(unet)
        load_model(unet, ema_unet, filename=checkpoint_path)
        print(f'the model is loaded from {checkpoint_path}')
        if use_ema:
            ema_unet.copy_to(unet.parameters())
        pipe = StableVideoDiffusionCILPipeline.from_pretrained(pretrained_model_path, unet=unet,scheduler=scheduler).to('cuda')
    else:
        pipe = StableVideoDiffusionCILPipeline.from_pretrained(pretrained_model_path,scheduler=scheduler).to(local_rank)

    # Get the training dataset
    dataset = ImageDataset(data_dir)
    distributed_sampler = DistributedSampler(
        dataset,
        shuffle=False,
    )

    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_processes,
        pin_memory=True,
        drop_last=False,
    )



    print(f'local_rank is {local_rank}')
    os.makedirs(save_dir, exist_ok=True)

    
 
    for step, batch in enumerate(dataloader):
        paths = batch['path']
        names = batch['name']
        
        images_list = load_PIL_images(paths, resolution)
        set_seed(seed)
        generator = torch.manual_seed(seed)
        samples = pipe(images_list, output_type="pt", generator=generator, height=resolution[1], width=resolution[0],
                num_inference_steps=num_step ,num_frames=num_frames,fps=fps,motion_bucket_id=motion_bucket_id,analytic_path=analytic_path
                    ).frames
        
        samples = torch.stack(samples)

        for sample, name in zip(samples, names):
            name = name.split('.')[0]
            save_path = f"{save_dir}/{name}.mp4"
            sample = rearrange(sample, "t c h w -> c t h w").unsqueeze(dim=0)
            save_videos_grid(sample.cpu(), save_path, n_rows=1)
            print(f'the sample has been saved in {save_path}')

def main(
    seed,
    resolution,
    batch_size,
    original_svd,
    use_ema,
    step,
    data_dir,
    motion_bucket_id,
    fps,
    pretrained_model_path,
    checkpoint_root,
    num_step,
    sigma_max,
    analytic_path=None
):


    # Initialize distributed training
    local_rank = init_dist(launcher="pytorch", backend="nccl")
    num_processes = dist.get_world_size()

   
    scheduler_path=f'./schedulers/scheduler_config{sigma_max}.json' 
    checkpoint_path = os.path.join(checkpoint_root, f'checkpoint-step-{int(step)}.ckpt') if not original_svd else None
   
    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        raise EOFError(f'the checkpoint {checkpoint_path} is nokbit existing')

    task_name =f'sigma_max_{sigma_max}'
    save_dir = os.path.join(f'results/inference',task_name)
    os.makedirs(save_dir, exist_ok=True)


    sampling(data_dir=data_dir,num_step=num_step,scheduler_path=scheduler_path,pretrained_model_path=pretrained_model_path,
             checkpoint_path=checkpoint_path, use_ema = use_ema,save_dir= save_dir, resolution=tuple(resolution), 
             batch_size=batch_size, num_processes=num_processes, local_rank=local_rank, 
             motion_bucket_id=motion_bucket_id,fps=fps,seed=seed,analytic_path=analytic_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,help='path to your config file')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(**config)

