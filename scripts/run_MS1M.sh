#! /bin/bash

## for webface42M 42M / 2048(batch size) = 20507 per epoch
## for MS1M-V3 5.2M / 512(batch size) = 2529 per epoch
#CUDA_VISIBLE_DEVICES=0 python dvae_main.py \
#--dataset "MS1M-V3" \
#--dset_dir "../MS1M-V3" \
#--dec_dist "gaussian" \
#--c_dim 20 \
#--beta 6.4 \
#--lr 0.0001 \
#--batch_size 64 \
#--image_size 112 \
#--seed 15 \
#--max_iter 8e5 \
#--log_line_iter 1000 \
#--log_img_iter 100000 \
#--ckpt_save_iter 100000 \
#--name "dvae_MS1M_V3" \
#--output_dir "outputs"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
--config "MS1M_V3-112.yaml" \
--dvae_name "dvae_celeba_aligned" \
--name "idgan_MS1M_V3" \
--output_dir "outputs" \
--seed 15
