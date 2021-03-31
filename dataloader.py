from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import multiprocessing as mt
import random
from skimage import util
from skimage import io
from skimage import transform
random.seed(0)

class Dataloader(object):
    def __init__(self, data_path, filenames_file, params, mode, num_output, sigma, noise_mode):
        self.params = params
        self.mode = mode
        self.num_output = num_output
        self.sigma = sigma
        self.noise_mode = noise_mode
        lists_and_labels = np.loadtxt(filenames_file, dtype=str)
        if self.mode == 'train':
            random.shuffle(lists_and_labels)
        image_path = lists_and_labels[:, :-2].tolist()
        pos_list = lists_and_labels[:, -2:].tolist()
        for i in range(len(image_path)):
            for j in range(len(image_path[i])):
                image_path[i][j] = os.path.join(data_path, image_path[i][j])
            for j in range(len(pos_list[i])):
                pos_list[i][j] = int(pos_list[i][j])

        total_num = len(lists_and_labels)

        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_path), tf.constant(pos_list)))
        if self.mode == 'train':
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=total_num, seed=0)
        dataset = dataset.map(self._parse_image, num_parallel_calls=mt.cpu_count())
        if self.mode == 'train':
            dataset = dataset.batch(self.params.batch_size)
        elif self.mode == 'test':
            dataset = dataset.batch(1)
        self.dataset = dataset

    def _parse_image(self, files_path, pos):
        pos = tf.cast(pos, dtype=tf.float32)
        images = tf.map_fn(self.process_one_image, files_path, dtype=tf.float32)
        images = tf.split(images, num_or_size_splits=10, axis=0)
        images = [tf.squeeze(image, axis=0) for image in images]

        # (h,w,27)
        input_imgs = tf.concat([images[i] for i in range(9)], axis=2)
        image_string = tf.read_file(files_path[-1])
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_converted = tf.image.convert_image_dtype(image_decoded, tf.float32)
        image_converted = image_converted[12:-12, 14:-15, :]
        label_img = tf.image.resize_images(image_converted, [self.params.height, self.params.width],
                                               tf.image.ResizeMethod.AREA)
        onehot_pos = self.pos_to_onehot(pos)
        return input_imgs, label_img, pos, onehot_pos

    def pos_to_onehot(self, pos):
        return tf.py_func(self.pos_to_onehot_py, [pos], tf.float32)

    def pos_to_onehot_py(self, pos):
        index = int(pos[1] * 7 + pos[0])
        onehot = np.zeros((self.params.height, 49, 1), dtype=np.float32)
        onehot[:, index, :] = 1
        return onehot

    def process_one_image(self, image_path):
        return tf.cast(self.read_input_img(image_path), dtype=tf.float32)


    def augment_image(self, image_list, num_output):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        side_image_aug = [image_list[i] ** random_gamma for i in range(num_output)]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        side_image_aug = [side_image_aug[i] * random_brightness for i in range(num_output)]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white_0 = tf.ones([tf.shape(image_list[0])[0], tf.shape(image_list[0])[1]])
        color_image_0 = tf.stack([white_0 * random_colors[i] for i in range(3)], axis=2)
        side_image_aug = [side_image_aug[i] * color_image_0 for i in range(num_output)]

        # saturate
        side_image_aug = [tf.clip_by_value(side_image_aug[i],  0, 1) for i in range(num_output)]

        return side_image_aug

    def adjustTone(self,image_list, num_output):
        img = [image_list[i] ** (1/1.5) for i in range(num_output)]
        img = [tf.image.rgb_to_hsv(img[i]) for i in range(num_output)]
        img = [img[i] * [1, 1.5, 1] for i in range(num_output)]
        img = [tf.image.hsv_to_rgb(img[i]) for i in range(num_output)]
        img = [tf.clip_by_value(img[i], 0, 1) for i in range(num_output)]
        return img

    def read_input_img(self, files_path):
        return tf.py_func(self.read_input, [files_path], tf.double)

    def read_input(self, files_path):
        img = io.imread(files_path.decode())

        # if self.params.height != 512:
        img = img[12:-12, 14:-15, :]
        img = transform.resize(img, (self.params.height, self.params.width))

        if self.noise_mode == 's&p':
            img = util.random_noise(img, 's&p', amount=self.sigma)
        if self.noise_mode == 'gauss':
            img = util.random_noise(img, 'gaussian', mean=0, var=(self.sigma/255)**2)
        return img
