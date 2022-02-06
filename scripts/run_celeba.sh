#! /bin/bash

# for webface42M 42M / 2048(batch size) = 20507 per epoch
#python dvae_main.py --dataset celeba --dec_dist gaussian --c_dim 20 --beta 6.4 --max_iter 1e5 --name dvae_celeba

#python dvae_main.py \
#--dataset "celeba_aligned" \
#--dset_dir "../celeba/celeba_arcface_aligned" \
#--dec_dist gaussian \
#--c_dim 20 \
#--beta 6.4 \
#--lr 0.0001 \
#--batch_size 64 \
#--image_size 112 \
#--seed 15 \
#--max_iter 2e5 \
#--name "dvae_celeba_aligned" \
#--output_dir "outputs"

#python train.py --config celebA-64.yaml --dvae_name dvae_celeba --name idgan_celeba_64

#python train.py \
#--config "celebA-112.yaml" \
#--dvae_name "dvae_celeba" \
#--name "idgan_celeba_112"

python train.py \
--config "celebA-112.yaml" \
--dvae_name "dvae_celeba_aligned" \
--name "idgan_celeba_aligned_112"

#python train.py --config celebA-256.yaml --dvae_name dvae_celeba --name idgan_celeba_256
#python train.py --config celebA-HQ.yaml --dvae_name dvae_celeba --name idgan_celeba_hq
