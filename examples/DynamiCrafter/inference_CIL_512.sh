seed=123

name=inference

ckpt=ckpt/finetuned/timenoise.ckpt # path to your checkpoint
config=configs/inference_512_v1.0.yaml

prompt_dir=prompts/512 # file for prompts, which includes images and their corresponding text
res_dir="results" # file for outputs


H=320
W=512
FS=24
M=1000

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=23459 --node_rank=0 \
scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width ${W} \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' \
--guidance_rescale 0.7 \
--perframe_ae \
--M ${M} \
--whether_analytic_init 1 \
--analytic_init_path 'ckpt/initial_noise_512.pt' 
 