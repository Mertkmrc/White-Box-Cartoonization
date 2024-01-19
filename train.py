'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''


import tensorflow as tf
import os
import argparse
import random
from tqdm import tqdm

from guided_filter import GuidedFilter
from dataset import MyTFDataset
import utils
import network
import loss
import numpy as np
random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--use_enhance", default = False)
    if os.getenv("COLAB_RELEASE_TAG"):
        parser.add_argument("--save_dir", default = 'results', type = str)
        parser.add_argument("--data-dir", default = '/content/drive/MyDrive/White-Box-Cartoonization/data/')
    else:
        parser.add_argument("--data-dir", default = 'data/')
        parser.add_argument("--save_dir", default = 'results_local', type = str)

    args = parser.parse_args()

    return args



def train(args):
    return 1


@tf.function
def train_step(sample_photo, sample_cartoon,d_optimizer, g_optimizer, disc_model, gen_model, guided_filter, total_iter,batch_idx):


    with tf.GradientTape() as tape_d:

        fake_cartoon = gen_model(sample_photo)
        fake_output = guided_filter.process(sample_photo, fake_cartoon, r=1)

        blur_fake = guided_filter.process(fake_output, fake_output, r=5, eps=2e-1)
        blur_cartoon = guided_filter.process(sample_cartoon, sample_cartoon, r=5, eps=2e-1)

        # utils.save_training_images(combined_image = sample_photo, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='normal')
        # utils.save_training_images(combined_image = fake_output, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='normal_out"')
        # utils.save_training_images(combined_image = sample_cartoon, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='cartoon"')
        # utils.save_training_images(combined_image = tf.concat((sample_photo, sample_cartoon), axis=2), step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='deneme_rep"')
        # utils.save_training_images(combined_image = tf.concat((sample_photo, sample_cartoon), axis=2), step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='deneme_rep"')
        # utils.save_training_images(combined_image = tf.concat((fake_cartoon, fake_output ), axis=2), step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='deneme_rep"')
        # utils.save_training_images(combined_image = tf.concat((sample_photo, fake_output), axis=2), step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='deneme_result"')

        gray_fake, gray_cartoon = utils.color_shift(fake_output, sample_cartoon)

        d_loss_gray, g_loss_gray = loss.lsgan_loss(disc_model, gray_cartoon, gray_fake)
        d_loss_blur, g_loss_blur = loss.lsgan_loss(disc_model, blur_cartoon, blur_fake)
        d_loss_total = (d_loss_blur + d_loss_gray)/ 2.0
        grads_d = tape_d.gradient(d_loss_total, disc_model.trainable_weights)
    d_optimizer.apply_gradients(zip(grads_d, disc_model.trainable_weights))

    with tf.GradientTape() as tape_g:

        fake_cartoon = gen_model(sample_photo)
        output = guided_filter.process(sample_photo, fake_cartoon, r=1)
        blur_fake = guided_filter.process(output, output, r=5, eps=2e-1)
        blur_cartoon = guided_filter.process(sample_cartoon, sample_cartoon, r=5, eps=2e-1)

        gray_fake, gray_cartoon = utils.color_shift(output, sample_cartoon)

        d_loss_gray, g_loss_gray = loss.lsgan_loss(disc_model, gray_cartoon, gray_fake )
        d_loss_blur, g_loss_blur = loss.lsgan_loss(disc_model, blur_cartoon, blur_fake )

        utils.save_training_images(combined_image = output, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='guided_filter_'+str(batch_idx))
        utils.save_training_images(combined_image = fake_cartoon, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='generator_output_'+str(batch_idx))
        utils.save_training_images(combined_image = sample_photo, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='input_normal_'+str(batch_idx))
        utils.save_training_images(combined_image = sample_cartoon, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='input_cartoon_'+str(batch_idx))

        if args.use_enhance:
            sample_superpixel = utils.selective_adacolor(output, power=1.2)
            test_superpixel = utils.selective_adacolor(sample_photo, power=1.2)
            utils.save_training_images(combined_image = test_superpixel, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='test_selectiveColor_superpixel_'+str(batch_idx))
        else:
            sample_superpixel = utils.simple_superpixel(output, seg_num=200)
            test_superpixel = utils.simple_superpixel(sample_photo, seg_num=200)
            sample_superpixel_converted = sample_superpixel.astype(np.float32)
            utils.save_training_images(combined_image = test_superpixel, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='test_simple_superpixel_'+str(batch_idx))
            utils.save_training_images(combined_image = sample_superpixel_converted, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='test_simple_superpixel_converted'+str(batch_idx))
        sample_superpixel_converted = sample_superpixel.astype(np.float32)

        vgg_model = loss.Vgg19(args.data_dir + 'vgg19_no_fc.npy')
        vgg_photo = vgg_model.build_conv4_4(sample_photo)
        vgg_output = vgg_model.build_conv4_4(output)
        vgg_superpixel = vgg_model.build_conv4_4(sample_superpixel_converted)
        h, w, c = vgg_photo.get_shape().as_list()[1:]


        photo_loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
        superpixel_loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference\
                                        (vgg_superpixel, vgg_output))/(h*w*c)

        recon_loss = photo_loss + superpixel_loss
        tv_loss = loss.total_variation_loss(output)
        # g_loss_total = tv_loss + g_loss_blur + g_loss_gray + recon_loss

        g_loss_total = 1e-4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e-2 * recon_loss
        grads_g = tape_g.gradient(g_loss_total, gen_model.trainable_weights)
    g_optimizer.apply_gradients(zip(grads_g, gen_model.trainable_weights))



    # blur_fake, gray_fake sample_superpixel,
    # sample_photo, fake_cartoon, output,

    representation_images = [blur_fake,gray_fake, sample_superpixel]

    cartoonization_images = [sample_photo, fake_cartoon, output]

    return g_loss_total, d_loss_total, recon_loss, representation_images, cartoonization_images
def main():
    tf.config.run_functions_eagerly(True)


    photo_dir, cartoon_dir = args.data_dir+'normal', args.data_dir+'cartoon'
    guided_filter = GuidedFilter()
    # face_photo_list = os.listdir(face_photo_dir)
    # face_cartoon_list = os.listdir(face_cartoon_dir_kyoto_face)
    # scenery_photo_list = os.listdir(photo_dir)
    # scenery_cartoon_list = os.listdir(cartoon_dir)
    my_dataset = MyTFDataset(photo_dir,cartoon_dir, 16)
    d_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002)
    g_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002)
    disc_sn_model = network.disc_sn()
    gen_model = network.UNetGenerator()

    # for total_iter in tqdm(range(args.total_iter)):
    for total_iter in tqdm(range(200)):
        print('='* 90)
        print('STARTED NEW ITER', total_iter)
        for batch_idx, (sample_photo, sample_cartoon) in enumerate(my_dataset):
            print('batch idx',batch_idx)
            g_loss, d_loss, recon_loss, rep_images, process_images = train_step(sample_photo, sample_cartoon, d_optimizer, g_optimizer,disc_sn_model, gen_model, guided_filter,total_iter, batch_idx)

        # if total_iter % 5 == 0:
        utils.save_training_images(combined_image = process_images[2], step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='guided_filter_output_')
        utils.save_training_images(combined_image = process_images[1], step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='generator_output_')
        utils.save_training_images(combined_image = sample_photo, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='input_normal_')
        utils.save_training_images(combined_image = sample_cartoon, step=total_iter,dest_folder=args.save_dir+'/images',suffix_filename='input_cartoon_')
        gen_model.save(args.save_dir+'/model/generator_'+str(total_iter)+'.keras')
        disc_sn_model.save(args.save_dir+'/model/discriminator'+str(total_iter)+'.keras')

        print('[Epoch: %d| - G loss: %.12f - D loss: %.12f - Recon loss: %.12f' % ((total_iter + 1), g_loss, d_loss, recon_loss))
    gen_model.save(args.save_dir+'/model/generator_'+str(total_iter)+'.keras')
    disc_sn_model.save(args.save_dir+'/model/discriminator'+str(total_iter)+'.keras')




if __name__ == '__main__':
    args = arg_parser()
    main()
