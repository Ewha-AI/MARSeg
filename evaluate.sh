##### chmod +x train.sh
##### ./train.sh

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="localhost" \
         --master_port=29500 \
         main_marseg.py \
         --model mar_base \
         --diffloss_d 6 \
         --diffloss_w 1024 \
         --diffusion_batch_mul 4 \
         --data_path ./data/msd/pancreas \
         --class_num 3 \
         --warmup_epochs 100 \
         --blr 2.5e-5 \
         --batch_size 32 \
         --epochs 200 \
         --layer_indices 8 9 10 11 \
         --mar_checkpoint_path ./pretrained_models/mar/checkpoint-399.pth \
         --vae_path ./pretrained_models/vae/kl16.ckpt \
         --output_dir ./output_dir/MSD_pancreas \
         --evaluate \
         --resume ./output_dir/MSD_pancreas \
         --checkpoint_name best.pth \
