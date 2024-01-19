'''
CVPR 2020 submission, Paper ID 6791
Source code for 'Learning to Cartoonize Using White-Box Cartoon Representations'
'''


import tensorflow as tf


class GuidedFilter(tf.Module):
    def box_filter(self, x, r):
        channel =  x.shape[3]  # Batch, H, W, Channel
        kernel_size = (2*r+1)
        weight = 1.0/(kernel_size**2)
        box_kernel = weight*tf.ones((2*r+1, 2*r+1, channel, 1), dtype=x.dtype)
        output = tf.nn.depthwise_conv2d(x, box_kernel, strides=[1, 1, 1, 1], padding='SAME')
        return output

    def guided_filter(self, x, y, r, eps=1e-2):
        # Batch, H, W, Channel
        _, H, W, _ = x.shape

        N = self.box_filter(tf.ones((1, H, W, 1), dtype=x.dtype), r)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
        var_x  = self.box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b
        return output

    def process(self, x, y, r, eps=1e-2):
        return self.guided_filter(x, y, r, eps)

if __name__ == '__main__':
    pass
