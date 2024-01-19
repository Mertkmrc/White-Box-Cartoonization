import tensorflow as tf

class UNetGenerator(tf.keras.Model):

    def __init__(self, channel=32, num_blocks=4, name='generator', padding=3,reuse=False):
        super().__init__()
        self.channel = channel
        self.num_blocks = num_blocks
        self.zero_padding3 = self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=2)
        self.zero_padding1 = self.zero_padding = tf.keras.layers.ZeroPadding2D(padding=3)
        self.conv1 = tf.keras.layers.Conv2D(channel, (7, 7),  padding='same',activation=None)
        self.conv2 = tf.keras.layers.Conv2D(channel, (3, 3), padding='same',strides=2,  activation=None)
        self.conv3 = tf.keras.layers.Conv2D(channel*2, (3, 3),padding='same',  activation=None)
        self.conv4 = tf.keras.layers.Conv2D(channel*2, (3, 3),padding='same', strides=2,  activation=None)
        self.conv5 = tf.keras.layers.Conv2D(channel*4, (3, 3),padding='same',  activation=None)
        self.conv6 = tf.keras.layers.Conv2D(channel*2, (3, 3),padding='same',  activation=None)
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv7 = tf.keras.layers.Conv2D(channel*2, (3, 3),padding='same',  activation=None)
        self.conv8 = tf.keras.layers.Conv2D(channel, (3, 3),  padding='same',activation=None)
        self.upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.conv9 = tf.keras.layers.Conv2D(channel, (3, 3),  padding='same',activation=None)
        self.conv10 = tf.keras.layers.Conv2D(3, (7, 7), padding='same', activation=None)
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.relu2 = tf.keras.layers.LeakyReLU()
        self.relu3 = tf.keras.layers.LeakyReLU()
        self.relu4 = tf.keras.layers.LeakyReLU()
        self.relu5 = tf.keras.layers.LeakyReLU()
        self.relu6 = tf.keras.layers.LeakyReLU()
        self.relu7 = tf.keras.layers.LeakyReLU()
        self.relu8 = tf.keras.layers.LeakyReLU()
        self.relu9 = tf.keras.layers.LeakyReLU()

    def resblock(self, inputs, out_channel=32, name='resblock'):

        x = tf.keras.layers.Conv2D(out_channel, (3, 3), padding='same',  activation=None)(inputs)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(out_channel, (3, 3), padding='same',  activation=None)(x)
        # x = tf.keras.layers.BatchNormalization()(x)

        return tf.keras.layers.add([x, inputs], name=name)

    def call(self, inputs):
        # x0 = self.zero_padding3(inputs)
        x0 = self.conv1(inputs)
        x0 = self.relu1(x0)

        # x0 = self.zero_padding1(x0)
        x1 = self.conv2(x0)
        x1 = self.relu2(x1)
        # x0 = self.zero_padding1(x1)
        x1 = self.conv3(x1)
        x1 = self.relu3(x1)

        # x1 = self.zero_padding1(x1)
        x2 = self.conv4(x1)
        x2 = self.relu4(x2)
        # x2 = self.zero_padding1(x2)
        x2 = self.conv5(x2)
        x2 = self.relu5(x2)
        for idx in range(self.num_blocks):
            x2 = self.resblock(x2, out_channel=self.channel*4, name='block_{}'.format(idx))

        x2 = self.conv6(x2)
        x2 = self.relu6(x2)

        x3 = self.upsample1(x2)
        x3 = self.conv7(tf.concat([x3, x1], axis=-1))
        x3 = self.relu7(x3)
        x3 = self.conv8(x3)
        x3 = self.relu8(x3)

        x4 = self.upsample2(x3)
        x4 = self.conv9(tf.concat([x4, x0], axis=-1))
        x4 = self.relu9(x4)
        x4 = self.conv10(x4)

        return x4

def disc_sn(channel=32, name='', reuse=False):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.Input(shape=(256, 256, 3)))
    for idx in range(3):
        model.add(tf.keras.layers.Conv2D(channel*2**idx, (3, 3), strides=2, name='conv{}_1'.format(idx)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))

        model.add(tf.keras.layers.Conv2D(channel*2**idx, (3, 3),  name='conv{}_2'.format(idx)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.3))


    model.add(tf.keras.layers.Conv2D(1, (1, 1), name='conv_out'.format(idx)))
    # if patch == True:
    #     model.add(tf.keras.layers.Conv2D(1, (1, 1), name='conv_out'.format(idx)))
    # else:
    #     model.add(tf.keras.layers.GlobalAveragePooling2D())
    #     model.add(tf.keras.layers.Dense(1, activation=None))

    return model

if __name__ == '__main__':
    pass

