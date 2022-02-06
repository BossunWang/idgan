import argparse
import os
from os import path
import copy
from tqdm import tqdm
import cv2
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from gan_training import utils
from gan_training.checkpoints import CheckpointIO
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models
)

import sys
sys.path.append('../FaceX-Zoo/')
from backbone.iresnet_insightface import iresnet100_insightface


def gen_image(zdist, evaluator, image, sample_size=1):
    ztest = zdist.sample((sample_size,))
    x_fake = evaluator.create_samples(ztest, image)
    x_mu_fake = evaluator.create_mu_samples(ztest, image)
    x_recon = evaluator.reconstruct(image)
    return x_fake, x_mu_fake, x_recon


def extract_feature(model, image):
    with torch.no_grad():
        feature = model(image).cpu().numpy()
    feature = np.squeeze(feature)
    feature = feature / np.linalg.norm(feature)
    return feature


def main(args):
    config = load_config(args.config)
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)

    # Shorthands
    out_dir = os.path.join(args.output_dir, args.name)

    config['training']['out_dir'] = out_dir
    out_dir = config['training']['out_dir']
    batch_size = config['test']['batch_size']
    checkpoint_dir = path.join(out_dir, 'chkpts')
    img_dir = "../MS1M-V3"

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0" if is_cuda else "cpu")

    dvae, generator, discriminator = build_models(config)
    dvae_ckpt_path = os.path.join(args.output_dir, config['dvae']['runname'], 'chkpts', config['dvae']['ckptname'])
    print("load dvae")
    print(dvae_ckpt_path)
    dvae_ckpt = torch.load(dvae_ckpt_path)['model_states']['net']
    dvae.load_state_dict(dvae_ckpt)

    # create FR model
    fr_model = iresnet100_insightface(112).to(device)
    model_path = "../FaceX-Zoo/pretrain_model/insightface_partial_fc/pytorch/partial_fc_glint360k_r100/16_backbone.pth"
    pretrained_dict = torch.load(model_path, map_location='cpu')
    fr_model.load_state_dict(pretrained_dict)
    fr_model.eval()

    # Put models on gpu if needed
    dvae = dvae.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Use multiple GPUs if possible
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        generator=generator,
        discriminator=discriminator,
    )

    # Distributions
    cdist = get_zdist('gauss', config['dvae']['c_dim'], device=device)
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Evaluator
    evaluator = Evaluator(generator, dvae, zdist,
                          batch_size=batch_size, device=device)

    # Load checkpoint if existant
    it = checkpoint_io.load('model.pt')

    ctest = cdist.sample((1,))
    ztest = zdist.sample((1,))
    x_fake = evaluator.create_random_samples(ztest, ctest)
    plt.imsave('sample_random.png', (x_fake.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    for dirPath, dirNames, fileNames in os.walk(img_dir):
        for f in fileNames:
            image_path = os.path.join(dirPath, f)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            BGR_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = (image.transpose((2, 0, 1)) - 127.5) / 127.5
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            image = torch.unsqueeze(image, 0).to(device)

            BGR_image = (BGR_image.transpose((2, 0, 1)) - 127.5) / 127.5
            BGR_image = BGR_image.astype(np.float32)
            BGR_image = torch.from_numpy(BGR_image)
            BGR_image = torch.unsqueeze(BGR_image, 0).to(device)

            x_fake1, x_mu_fake1, x_recon1 = gen_image(zdist, evaluator, image, sample_size=1)
            x_fake2, x_mu_fake2, x_recon2 = gen_image(zdist, evaluator, image, sample_size=1)

            org_feature = extract_feature(fr_model, BGR_image)
            gen_feature1 = extract_feature(fr_model, x_fake1)
            gen_feature2 = extract_feature(fr_model, x_fake2)

            cos_score1 = np.dot(org_feature, gen_feature1)
            cos_score2 = np.dot(org_feature, gen_feature2)
            cos_score3 = np.dot(gen_feature1, gen_feature2)
            print("cos_score1:", cos_score1)
            print("cos_score2:", cos_score2)
            print("cos_score3:", cos_score3)

            sample_generate_sample1 = torch.cat((image[0], x_fake1[0]), 2)
            sample_generate_mu_sample1 = torch.cat((image[0], x_mu_fake1[0]), 2)
            sample_rec_sample1 = torch.cat((image[0], x_recon1[0]), 2)
            plt.imsave('sample_generate_sample1.png', (sample_generate_sample1.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave('sample_generate_mu_sample1.png', (sample_generate_mu_sample1.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave('sample_rec_sample1.png', (sample_rec_sample1.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

            sample_generate_sample2 = torch.cat((image[0], x_fake2[0]), 2)
            sample_generate_mu_sample2 = torch.cat((image[0], x_mu_fake2[0]), 2)
            sample_rec_sample2 = torch.cat((image[0], x_recon2[0]), 2)
            plt.imsave('sample_generate_sample2.png',
                       (sample_generate_sample2.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave('sample_generate_mu_sample2.png',
                       (sample_generate_mu_sample2.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave('sample_rec_sample2.png', (sample_rec_sample2.cpu().numpy().transpose(1, 2, 0) + 1) / 2)


    # # Inception score
    # if config['test']['compute_inception']:
    #     print('Computing inception score...')
    #     inception_mean, inception_std = evaluator.compute_inception_score()
    #     print('Inception score: %.4f +- %.4f' % (inception_mean, inception_std))
    #
    # # Samples
    # print('Creating samples...')
    # ztest = zdist.sample((sample_size,))
    # x = evaluator.create_samples(ztest)
    # utils.save_images(x, path.join(img_all_dir, '%08d.png' % it),
    #                   nrow=sample_nrow)
    # if config['test']['conditional_samples']:
    #     for y_inst in tqdm(range(nlabels)):
    #         x = evaluator.create_samples(ztest, y_inst)
    #         utils.save_images(x, path.join(img_dir, '%04d.png' % y_inst),
    #                           nrow=sample_nrow)`


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Test a trained GAN and create visualizations.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='Path to outputs directory')
    parser.add_argument('--name', type=str, help='Name of the experiment')

    args = parser.parse_args()

    main(args)