import torch
import os
from einops import rearrange
import subprocess
import torchvision
import imageio
import numpy as np
import torch.distributed as dist
from PIL import Image
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

def zero_rank_print(s):
    if dist.get_rank() == 0:
        print("### " + s)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def load_PIL_images(paths, resize_to=None):
    '''
     Loads `image` to PIL Images.
    :param paths: image path list
    :param resize_to: the resize resolution
    '''
    images = []
    for path in paths:
        img = Image.open(path)
        # 如果指定了 resize_to，调整图像大小
        if resize_to:
            img = img.resize(resize_to)
        images.append(img)
    return images

def save_image(img: torch.Tensor, path, rescale=False):
    """
    save images
    Args:
        img: the image for saving with shape (C,H,W) with range [-1, 1] or [0, 1]
        path: the path for save image
        rescale: whether rescale img to [0, 1]
    """
    if rescale:
        img = (img + 1.0) / 2.0  # -1,1 -> 0,1
    img = img * 255 #[-1,1] to {0,……,255}
    img = rearrange(img.cpu().numpy(), 'C H W -> H W C')
    Image.fromarray(img.astype(np.uint8)).save(path)

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK']) # global rank
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

    return local_rank

import random
def set_seed(seed=1234):
    """
    set seed in "random", "numpy" and "torch" for reproducible

    Args:
        seed: the seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_checkpoint(model, optimizer, ema, filename):
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    if 'ema_state' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state'])
    return checkpoint['global_step']

def save_checkpoint(output_dir, unet, optimizer, global_step, ema_unet):
    save_path = os.path.join(output_dir, f"checkpoints")
    model_state_dict = unet.state_dict()
    state_dict = {
        "global_step": global_step,
        "state_dict": model_state_dict,
        'optimizer': optimizer.state_dict(),
        "ema_state": ema_unet.state_dict()
    }
    torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
    print(f"Saved state to {save_path} (global_step: {global_step})")


def initialize_distributed(type):
    if type == 'ddp':
        device = init_dist(launcher="pytorch", backend='nccl')
        global_rank = dist.get_rank()
        is_main_process = global_rank == 0
        accelerator = None
    else:
        raise ValueError(f"unknown initialize distribution: {type}")
    return accelerator, device, is_main_process

from importlib import import_module
def get_instance(config, from_pretrained):
    class_path = config['type']
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    class_instance = getattr(module, class_name)
    params = config.get('params', {})

    if from_pretrained:
        instance = class_instance.from_pretrained(**params)
    else:
        instance = class_instance(**params)
    return instance

def get_model(model, device):
    feature_extractor = get_instance(model["feature_extractor"])
    image_encoder = get_instance(model["image_encoder"])
    vae = get_instance(model["vae"])
    unet = get_instance(model["unet"])

    vae.to(device)
    image_encoder.to(device)
    unet.to(device)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    return feature_extractor, image_encoder, vae, unet

def get_optimizer(lr, unet, trainable_modules=None):
    if trainable_modules is None:
        params = unet.parameters()
    else:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        params = list(filter(lambda p: p.requires_grad, unet.parameters()))

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    return optimizer

def get_pipeline(config, **params):
    class_path = config['type']
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    class_instance = getattr(module, class_name)
    config_params = config.get('params', {})
    pipeline = class_instance.from_pretrained(**config_params, **params).to("cuda")
    return pipeline

def get_wrapper(type, unet, optimizer, device, accelerator=None):
    if type == 'ddp':
        unet = DDP(unet, device_ids=[device], output_device=device)
    else:
        raise ValueError(f"unknown distribution: {type}")
    return unet, optimizer

def set_logger(path, file_path=None):
    os.makedirs(path, exist_ok=True)
    # logger to print information
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    if file_path is not None:
        handler2 = logging.FileHandler(os.path.join(path, file_path), mode='w')
    else:
        handler2 = logging.FileHandler(os.path.join(path, "logs.txt"), mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

def create_output(output_dir, config):
    now = datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(output_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    writer = SummaryWriter(f"{output_dir}/logs")
    set_logger(output_dir, 'logs_' + now + '.txt')
    return writer, output_dir

