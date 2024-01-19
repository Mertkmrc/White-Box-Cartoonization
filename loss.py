

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:

    def __init__(self, vgg19_npy_path=None):
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()


    def build_conv4_4(self, rgb):

        rgb_scaled = (rgb+1) * 127.5

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                        green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.pool1 = self.max_pool(self.relu1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.pool2 = self.max_pool(self.relu2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
        self.relu3_4 = tf.nn.relu(self.conv3_4)
        self.pool3 = self.max_pool(self.relu3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
        self.relu4_4 = tf.nn.relu(self.conv4_4)
        self.pool4 = self.max_pool(self.relu4_4, 'pool4')

        return self.conv4_4

    def max_pool(self, bottom, name):
        max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME', name=name)(bottom)
        return max_pooling


    def conv_layer(self, bottom, name, kernel_size=(3, 3)):
        filters = self.get_conv_filter(name)

        conv = tf.nn.conv2d(input=bottom,filters=filters, strides=(1, 1), padding='SAME', name=name)
        conv_biases = self.get_bias(name)

        bias = tf.nn.bias_add(conv, conv_biases)
        return bias



    def fc_layer(self, bottom, name):
        shape = bottom.shape.as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        fc = tf.keras.layers.Dense(units=dim, name=name)(x)
        return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def lsgan_loss(discriminator, real, fake):
    real_logit = discriminator(real)
    fake_logit = discriminator(fake)

    g_loss = tf.reduce_mean(input_tensor=(fake_logit - 1)**2)
    d_loss = 0.5*(tf.reduce_mean(input_tensor=(real_logit - 1)**2) + tf.reduce_mean(input_tensor=fake_logit**2))
    return d_loss, g_loss



def total_variation_loss(image, k_size=1):
    h, w = image.get_shape().as_list()[1:3]
    tv_h = tf.reduce_mean(input_tensor=(image[:, k_size:, :, :] - image[:, :h - k_size, :, :])**2)
    tv_w = tf.reduce_mean(input_tensor=(image[:, :, k_size:, :] - image[:, :, :w - k_size, :])**2)
    tv_loss = (tv_h + tv_w)/(3*h*w)
    return tv_loss




if __name__ == '__main__':
    pass


