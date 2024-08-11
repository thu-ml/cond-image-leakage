import argparse
import math
import os
import json
import copy
os.environ['CUDA_VISIBILE_DEVICES']='1'
from typing import Dict, Optional
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import numpy as np
from PIL import Image
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DDPMScheduler
# DDIMScheduler Line 337,add timesteps= timesteps.clip(0,self.config.num_train_timesteps-1) to avoid -1,which will destroy the generation
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.models.attention import BasicTransformerBlock
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange
import imageio
from models.unet_3d_condition_mask import UNet3DConditionModel
from models.pipeline import LatentToVideoPipeline
from utils.common import calculate_motion_precision, calculate_latent_motion_score,DDPM_forward_timesteps,tensor_to_vae_latent


def load_primary_models(pretrained_model_path, in_channels=-1, motion_strength=False):
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    if in_channels>0 and unet.config.in_channels != in_channels:
        #first time init, modify unet conv in
        unet2 = unet
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", 
            in_channels=in_channels,
            low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True, 
            motion_strength=motion_strength)
        unet.conv_in.bias.data = copy.deepcopy(unet2.conv_in.bias)
        torch.nn.init.zeros_(unet.conv_in.weight)
        load_in_channel = unet2.conv_in.weight.data.shape[1]
        unet.conv_in.weight.data[:,in_channels-load_in_channel:]= copy.deepcopy(unet2.conv_in.weight.data)
        del unet2

    return noise_scheduler, tokenizer, text_encoder, vae, unet
      
def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def cast_to_gpu_and_type(model_list, device, weight_dtype):
    for model in model_list:
        if model is not None: model.to(device, dtype=weight_dtype)


def eval(pipeline, vae_processor, validation_data,validation_data_prompt,validation_data_prompt_image, out_file, index, forward_t=25, preview=True,mask_nodes=0):
    vae = pipeline.vae
    diffusion_scheduler =  pipeline.scheduler
    device = vae.device
    dtype = vae.dtype

    prompt = validation_data_prompt
    pimg = Image.open(validation_data_prompt_image)
    if pimg.mode == "RGBA":
        pimg = pimg.convert("RGB")
    pimg = T.Resize((384,512), antialias=False)(pimg)
    pimg=T.CenterCrop((validation_data.height, validation_data.width))(pimg)
    width, height = pimg.size
    scale = math.sqrt(width*height / (validation_data.height*validation_data.width))
    validation_data.height = round(height/scale/8)*8
    validation_data.width = round(width/scale/8)*8
    input_image = vae_processor.preprocess(pimg, validation_data.height, validation_data.width)
    input_image = input_image.unsqueeze(0).to(dtype).to(device)
    input_image_latents = tensor_to_vae_latent(input_image, vae)

    
    if 'mask' in validation_data:
        mask = Image.open(validation_data.mask)
        mask = mask.resize((validation_data.width, validation_data.height))
        np_mask = np.array(mask)
        np_mask[np_mask!=0]=255
    else:
        np_mask = np.ones([validation_data.height, validation_data.width], dtype=np.uint8)*255
    # out_mask_path = os.path.splitext(out_file)[0] + "_mask.jpg"
    # Image.fromarray(np_mask).save(out_mask_path)
    initial_latents, timesteps = DDPM_forward_timesteps(input_image_latents, forward_t, validation_data.num_frames, diffusion_scheduler) 
    mask = T.ToTensor()(np_mask).to(dtype).to(device)
    b, c, f, h, w = initial_latents.shape
    mask = T.Resize([h, w], antialias=False)(mask)
    mask = rearrange(mask, 'b h w -> b 1 1 h w')
    motion_strength = validation_data.get("strength", index+3)
    with torch.no_grad():
        video_frames, video_latents = pipeline(
            prompt=prompt,
            latents=initial_latents,
            width=validation_data.width,
            height=validation_data.height,
            num_frames=validation_data.num_frames,
            num_inference_steps=validation_data.num_inference_steps,
            guidance_scale=validation_data.guidance_scale,
            condition_latent=input_image_latents,
            mask=mask,
            motion=[motion_strength],
            return_dict=False,
            timesteps=timesteps,
            mask_nodes=mask_nodes
        )
    if preview:
        fps = validation_data.get('fps', 8)
        #imageio.mimwrite(out_file, video_frames, duration=int(1000/fps), loop=0)
        imageio.mimwrite(out_file.replace('gif', 'mp4'), video_frames, fps=fps)
    real_motion_strength = calculate_latent_motion_score(video_latents).cpu().numpy()[0]
    precision = calculate_motion_precision(video_frames, np_mask)
    print(f"save file {out_file.replace('gif', 'mp4')}, motion strength {motion_strength} -> {real_motion_strength}, motion precision {precision}")

    del pipeline
    torch.cuda.empty_cache()
    return precision

def batch_eval(unet, text_encoder, vae, vae_processor, pretrained_model_path, 
    validation_data,validation_data_prompt,validation_data_prompt_image, output_dir, preview, global_step=0, iters=1,mask_nodes=0,M=1000):
    device = vae.device
    dtype = vae.dtype
    unet.eval()
    text_encoder.eval()
    pipeline = LatentToVideoPipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet
    )
    pipeline.scheduler.config['num_train_timesteps']=M
    diffusion_scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)
    pipeline.scheduler = diffusion_scheduler

    motion_errors = []
    motion_precisions = []
    motion_precision = 0
    for t in range(iters):
        name= os.path.basename(validation_data_prompt_image)
        out_file_dir = output_dir
        os.makedirs(out_file_dir, exist_ok=True)
        out_file = f"{out_file_dir}/{name.split('.')[0]}.gif"
        precision = eval(pipeline, vae_processor, 
            validation_data, validation_data_prompt,validation_data_prompt_image,out_file, t, forward_t=validation_data.num_inference_steps, preview=preview,mask_nodes=mask_nodes)
        motion_precision += precision
    motion_precision = motion_precision/iters
    print(validation_data_prompt_image, "precision", motion_precision)
    del pipeline

def main_eval(
    pretrained_model_path: str,
    validation_data: Dict,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = 4444,
    motion_mask = False, 
    motion_strength = False,
    mask_nodes=0,
    M=1000,
    name='',
    **kwargs
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path, motion_strength=motion_strength)
    vae_processor = VaeImageProcessor()
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])
    
    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.half

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, unet, vae]
    cast_to_gpu_and_type(models_to_cast, torch.device("cuda"), weight_dtype)
    
    with open(validation_data.dataset_jsonl, 'r', encoding='utf-8') as a_file:
        for line in a_file:
            validationer = json.loads(line)  
            validation_data_prompt_image = validationer.get('dir', '')  
            validation_data_prompt = validationer.get('text', '')  
            batch_eval(unet, text_encoder, vae, vae_processor, pretrained_model_path, 
                validation_data,validation_data_prompt,validation_data_prompt_image, f"output/{name}/{M}", True,mask_nodes=mask_nodes,M=M)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    if args.eval:
        main_eval(**args_dict)

