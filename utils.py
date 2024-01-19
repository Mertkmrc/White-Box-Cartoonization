from skimage import segmentation
from joblib import Parallel, delayed
from selective_search.util import switch_color_space
from selective_search.structure import HierarchicalGrouping

import os
import numpy as np
import tensorflow as tf
from skimage.color import label2rgb
from keras.preprocessing.image import save_img
import re


from PIL import Image


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = tf.split(image1, num_or_size_splits=3, axis=3)
    b2, g2, r2 = tf.split(image2, num_or_size_splits=3, axis=3)

    if mode == 'normal':
        b_weight = tf.random.normal(shape=[3], mean=0.114, stddev=0.1)
        g_weight = np.random.normal(shape=[3], mean=0.587, stddev=0.1)
        r_weight = np.random.normal(shape=[3], mean=0.299, stddev=0.1)
    elif mode == 'uniform':
        b_weight = tf.random.uniform(shape=[3], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[3], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[3], minval=0.199, maxval=0.399)
    output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
    output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
    return output1, output2

def color_ss_map(image, seg_num=200, power=1,
                 color_space='Lab', k=10, sim_strategy='CTSF'):

    img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping

    while S.num_regions() > seg_num:

        i,j = S.get_highest_similarity()
        S.merge_region(i,j)
        S.remove_similarities(i,j)
        S.calculate_similarity_for_new_region()

    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image+1)/2
    image = image**power
    image = image/np.max(image)
    image = image*2 - 1

    return image

def selective_adacolor(batch_image, seg_num=200, power=1):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)\
                         (image, seg_num, power) for image in batch_image)
    return np.array(batch_out)

# TODO: replace skimage.color.label2rgb with the below function. This function gets error for
# def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):
#     out = np.zeros_like(image)
#     labels = np.unique(label_field)
#     bg = (labels == bg_label)
#     if bg.any():
#         labels = labels[labels != bg_label]
#         mask = (label_field == bg_label).nonzero()
#         out[mask] = bg_color
#     for label in labels:
#         mask = (label_field == label).nonzero()
#         color: np.ndarray = None
#         if kind == 'avg':
#             color = image[mask].mean(axis=0)
#         elif kind == 'median':
#             color = np.median(image[mask], axis=0)
#         elif kind == 'mix':
#             std = np.std(image[mask])
#             if std < 20:
#                 color = image[mask].mean(axis=0)
#             elif 20 < std < 40:
#                 mean = image[mask].mean(axis=0)
#                 median = np.median(image[mask], axis=0)
#                 color = 0.5 * mean + 0.5 * median
#             elif 40 < std:
#                 color = np.median(image[mask], axis=0)
#         out[mask] = color
#     return out

def slic(image, seg_num=200, kind='avg'):
    seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,compactness=10, convert2lab=True)
    image = label2rgb(seg_label, image, kind=kind, bg_label=-1)
    return image

def simple_superpixel(batch_image, seg_num=200, kind='avg'):
    num_job = np.shape(batch_image)[0]
    out = Parallel(n_jobs=num_job)(delayed(slic)\
                         (image, seg_num, kind) for image in batch_image)
    # segments = segmentation.slic(batch_image, n_segments=200, compactness=10)
    # out = label2rgb(segments, batch_image, kind='avg')
    np_out = np.array(out,dtype=type(batch_image))
    return np_out

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return None

def save_training_images(combined_image,  step, dest_folder, suffix_filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    image_path = os.path.join(dest_folder, f"{suffix_filename}_step_{step}.png")
    arr = np.reshape(combined_image, (4,4,256,256,3))
    arr_h = np.concatenate([arr[i,:,:,:,:] for i in range(4)], axis=2)
    arr_v = np.concatenate([arr_h[i,:,:,:] for i in range(4)], axis=0)
    save_img(image_path, arr_v)


if __name__ == '__main__':
    pass


