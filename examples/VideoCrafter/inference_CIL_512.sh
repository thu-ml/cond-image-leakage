ckpt='ckpt/model.ckpt' # path to your checkpoint
config='configs/inference_i2v_512_v1.0.yaml'

prompt_file="prompts/512/test_prompts.txt" # file for 
condimage_dir="prompts/512" # file for conditional images
res_dir="results" # file for outputs

H=320
W=512
FS=24
M=1000

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=23456 --node_rank=0 \
scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed 123 \
--mode 'i2v' \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir \
--n_samples 1 \
--bs 1 --height ${H} --width ${W} \
--unconditional_guidance_scale 12.0 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_file $prompt_file \
--cond_input $condimage_dir \
--fps ${FS} \
--savefps 8 \
--frames 16 \
--M ${M} \
--analytic_init_path "ckpt/initial_noise.pt"
