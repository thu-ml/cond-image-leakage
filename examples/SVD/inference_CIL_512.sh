CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --master_port=12345 --nproc_per_node=6 inference.py --config "config/inference512.yaml"
