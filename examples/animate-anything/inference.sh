torchrun --nproc_per_node=1 --master_port=29502 inference.py \
    --config output/latent/animate_anything_512_v1.02/config.yaml \
    --eval \
    M=900 \
    validation_data.dataset_jsonl="demo/demo.jsonl"