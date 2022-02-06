#! /bin/bash

# for webface42M 42M / 2048(batch size) = 20507 per epoch
python dvae_main.py \
--dataset umdface \
--dec_dist gaussian \
--c_dim 20 \
--beta 6.4 \
--max_iter 2e5 \
--name dvae_umdface \
--dset_dir "../Umdfaces/images"
#python train.py --config celebA-64.yaml --dvae_name dvae_celeba --name idgan_celeba_64
#python train.py \
#--config "celebA-112.yaml" \
#--dvae_name "dvae_celeb" \
#--name "idgan_celeba_112"
#python train.py --config celebA-256.yaml --dvae_name dvae_celeba --name idgan_celeba_256
#python train.py --config celebA-HQ.yaml --dvae_name dvae_celeba --name idgan_celeba_hq
