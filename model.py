# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple
import tensorflow.contrib.slim as slim
from bilinear_sampler import *
from bilinear_sampler_y import *

parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary,'
                        'num_gpus')

class Model(object):

    def __init__(self, params, mode, input_image, input_orig_image,label_image, target_pos, target_onehot_pos, reuse_variables=None, model_index=0, num_output=1):
        self.params = params
        self.mode = mode
        self.label_image = label_image
        self.input_image = input_image
        self.onehot = target_onehot_pos
        self.input_label = input_orig_image

        self.target_pos = target_pos
        self.num_output = num_output
        self.model_collection = ['model_' + str(model_index)]
        self.reuse_variables = reuse_variables
        self.img_input_pos = tf.constant([[0, 0], [3, 0], [6, 0],
                                          [0, 3], [3, 3], [6, 3],
                                          [0, 6], [3, 6], [6, 6]], dtype=tf.float32)
        batch_pos = []
        for batch in range(input_image.get_shape().as_list()[0]):
            pos_x = tf.ones((1, self.params.height, self.params.width, 1)) * (self.target_pos[batch][0] / 6)
            pos_y = tf.ones((1, self.params.height, self.params.width, 1)) * (self.target_pos[batch][1] / 6)
            batch_pos.append(tf.concat([pos_x, pos_y], axis=3))
        self.all_pos = tf.concat(batch_pos, axis=0)
        with tf.variable_scope('model', reuse=self.reuse_variables):
            self.all_disp = self.build_resnet50(self.input_image)
            self.disp = tf.matmul(self.all_disp, self.onehot)

            self.denoise_input = self.input_network(self.input_image, output_channel=9*3)
            self.wraped_image = self.wrap_all_image(self.denoise_input, self.disp, self.target_pos)
            self.color_output = self.build_color_net(self.wraped_image, self.disp)
        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def generate_image(self, img, disp, x_offset, y_offset, mode=""):
        if mode == "side_to_center":
            x_offset = -x_offset
            y_offset = -y_offset
        img = tf.cond(tf.equal(x_offset, tf.constant(value=0, dtype=tf.float32)), lambda: img, lambda: bilinear_sampler_1d_h(img, disp * x_offset))
        img = tf.cond(tf.equal(y_offset, tf.constant(value=0, dtype=tf.float32)), lambda: img, lambda: bilinear_sampler_1d_w(img, disp * y_offset))
        return img

    def wrap_all_image(self, batch_img, batch_disp, batch_target_pos):
        with tf.variable_scope('images'):
            batchsize = batch_img.get_shape().as_list()[0]
            all_wraped_list = []
            for batch in range(batchsize):
                target_x = tf.cast(batch_target_pos[batch][0], tf.float32)
                target_y = tf.cast(batch_target_pos[batch][1], tf.float32)
                disp = tf.expand_dims(batch_disp[batch, :, :, :], axis=0)
                wraped_list = []
                for i in range(9):
                    img = tf.expand_dims(batch_img[batch, :, :, i*3:(i+1)*3], axis=0)
                    x_offset = tf.cast(self.img_input_pos[i][0] - target_x, dtype=tf.float32)
                    y_offset = tf.cast(self.img_input_pos[i][1] - target_y, dtype=tf.float32)
                    wraped_img = self.generate_image(img=img, disp=disp, x_offset=x_offset, y_offset=y_offset, mode='side_to_center')
                    wraped_list.append(wraped_img)
                all_wraped_list.append(tf.concat(wraped_list, axis=3))
            return tf.concat(all_wraped_list, axis=0)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disp(self, x, num_out):
        disp = (self.conv(x, num_out, 3, 1, tf.nn.sigmoid)-0.5) * 8
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_resnet50(self, input):
        resnet_output = 49 * self.num_output
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            #set convenience functions
            conv   = self.conv
            if self.params.use_deconv:
                upconv = self.deconv
            else:
                upconv = self.upconv

            with tf.variable_scope('encoder'):
                conv1 = conv(input, 64, 7, 2) # H/2  -   64D
                pool1 = self.maxpool(conv1,           3) # H/4  -   64D
                conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
                conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
                conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = pool1
                skip3 = conv2
                skip4 = conv3

            # DECODING
            with tf.variable_scope('decoder'):

                upconv5 = upconv(conv4, 256, 3, 2) #H/16
                concat5 = tf.concat([upconv5, skip4], 3)
                iconv5  = conv(concat5,   256, 3, 1)

                upconv4 = upconv(iconv5,  128, 3, 2) #H/8
                concat4 = tf.concat([upconv4, skip3], 3)
                iconv4  = conv(concat4,   128, 3, 1)
                disp4 = self.get_disp(iconv4, resnet_output)
                udisp4  = self.upsample_nn(disp4, 2)

                upconv3 = upconv(iconv4,   64, 3, 2) #H/4
                concat3 = tf.concat([upconv3, skip2, udisp4], 3)
                iconv3  = conv(concat3,    64, 3, 1)
                disp3 = self.get_disp(iconv3, resnet_output)
                udisp3  = self.upsample_nn(disp3, 2)

                upconv2 = upconv(iconv3,   32, 3, 2) #H/2
                concat2 = tf.concat([upconv2, skip1, udisp3], 3)
                iconv2  = conv(concat2,    32, 3, 1)
                disp2 = self.get_disp(iconv2, resnet_output)
                udisp2  = self.upsample_nn(disp2, 2)

                upconv1 = upconv(iconv2,  16, 3, 2) #H
                concat1 = tf.concat([upconv1, udisp2], 3)
                iconv1  = conv(concat1,   16, 3, 1)
                return self.get_disp(iconv1, resnet_output)

    def input_network(self, x, output_channel):
        input = x

        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)

        x = slim.conv2d(x, output_channel, 3, padding='same', activation_fn=None)
        x = x + input
        return x

    def color_network(self, x, output_channel):
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 64, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 32, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 32, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, 16, 3, padding='same', activation_fn=tf.nn.elu)
        x = slim.conv2d(x, output_channel, 3, padding='same', activation_fn=tf.nn.sigmoid)
        return x


    def build_color_net(self, input, disp):
        color_net_input = tf.concat([input, self.all_pos, self.denoise_input], axis=3)
        with tf.variable_scope('color_net', reuse=self.reuse_variables):
            # (batch , h, w, 3)
            return self.color_network(color_net_input, self.num_output * 3)

    def image_loss(self, output, label):
        self.l1_im = tf.reduce_mean(tf.abs(output - label))
        self.ssim_im = tf.reduce_mean(self.SSIM(output, label))
        image_loss = (1-self.params.alpha_image_loss)*self.l1_im + self.params.alpha_image_loss*self.ssim_im
        return image_loss


    def build_losses(self):
        with tf.variable_scope('losses'):
            self.ssim_loss = (1 - tf.reduce_mean(tf.image.ssim(self.color_output, self.label_image, max_val=1.0)))/2
            self.l1_loss = tf.reduce_mean(tf.abs(self.color_output-self.label_image))

            self.warped_image_loss = tf.add_n([tf.reduce_mean(tf.abs(self.wraped_image[:, :, :, i*3:(i+1)*3] - self.label_image))
                                               for i in range(9)])/9
            self.total_loss = (self.ssim_loss + self.l1_loss + self.warped_image_loss) * 49

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.image('disp', (self.disp + 4.0) / 8, collections=self.model_collection)
            tf.summary.image('img_center_input', (self.input_image[:, :, :, 12:15]), collections=self.model_collection)
            tf.summary.image('denoise_img_center_input', (self.denoise_input[:, :, :, 12:15]), collections=self.model_collection)
            tf.summary.image('input_label', (self.input_label[:, :, :, 12:15]), collections=self.model_collection)
            tf.summary.image('warped_img', (self.wraped_image[:, :, :, 12:15]), collections=self.model_collection)
            tf.summary.image('output', (self.color_output), collections=self.model_collection)
            tf.summary.image('gt', (self.label_image), collections=self.model_collection)
            tf.summary.scalar('image_loss', self.ssim_loss, collections=self.model_collection)
            tf.summary.scalar('wraped_image_loss', self.warped_image_loss, collections=self.model_collection)
            tf.summary.scalar('image_l2_loss', self.l1_loss, collections=self.model_collection)


    def get_disparity_smoothness(self, disp, image):
        def gradient_x(img):
            gx = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gx

        def gradient_y(img):
            gy = img[:, :-1, :, :] - img[:, 1:, :, :]
            return gy

        disp_gradients_x = gradient_x(disp)
        disp_gradients_y = gradient_y(disp)


        image_gradients_x = gradient_x(image)
        image_gradients_y = gradient_y(image)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = tf.abs(disp_gradients_x * weights_x)
        smoothness_y = tf.abs(disp_gradients_y * weights_y)
        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
