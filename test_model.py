
# -*- coding: utf-8 -*-

# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com
from __future__ import absolute_import, division, print_function
from model import *
from dataloader import *
from average_gradients import *
import argparse
import time
from PIL import Image
from skimage import measure
# only keep warnings and errors
import os
from skimage import color
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='NoisyLFRecon TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='training or test', default='test')
parser.add_argument('--model_name',                type=str,   help='model name', default='NoisyLFRecon')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='resnet50')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, HCI, or Stanford', default='Stanford')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='./')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames txt file', default='./')
parser.add_argument('--test_file',                 type=str,   help='path to the test file',           default='./TCW_test_set.txt')
parser.add_argument('--input_height',              type=int,   help='input height', default=352)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=3)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', default=False)
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=2)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=4)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='./output')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='./ckpt')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', default=True)
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', default=False)
parser.add_argument('--num_output',                type=int,   help='the number of output', default=1)
args = parser.parse_args()


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)



def test(params, step=0, sigma=0, mode=''):
    """Test function."""
    print(step)
    dataloader = Dataloader(data_path=args.data_path,
                            filenames_file=args.test_file,
                            params=params,
                            mode='test',
                            num_output=args.num_output,
                            sigma=sigma,
                            noise_mode=mode
                            )
    dataset = dataloader.dataset
    input_images, label_image, target_pos, onehot_pos = dataset.make_one_shot_iterator().get_next()

    input_images.set_shape([1, params.height, params.width, 27])
    label_image.set_shape([1, params.height, params.width, 3])
    target_pos.set_shape([1, 2])
    onehot_pos.set_shape([1, params.height, 49, 1])

    reuse_variables = tf.AUTO_REUSE
    model = Model(params=params,
                  mode='test',
                  input_image=input_images,
                  input_orig_image=None,
                  label_image=label_image,
                  target_pos=target_pos,
                  target_onehot_pos=onehot_pos,
                  reuse_variables=reuse_variables,
                  num_output=args.num_output
                  )

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver(max_to_keep=30)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        # restore_path = args.checkpoint_path.split(".")[0]
        restore_path = args.checkpoint_path + '/model-{}'.format(step)
        print("restore from {}".format(restore_path))
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.test_file) // 49
    print('Now evalue {} samples'.format(num_test_samples))

    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory

    for num in range(num_test_samples):
        print('Sample No.' + str(num + 1))

        path_to_disp_est = output_directory + '/' + args.model_name + '/model_%s_%f' % (mode,sigma) + '/disp_est' + '/sample_%04d' % num
        path_to_img_est = output_directory + '/' + args.model_name + '/model_%s_%f' % (mode,sigma) + '/est_img' + '/sample_%04d' % num
        path_to_gt = output_directory + '/' + args.model_name + '/model_%s_%f' % (mode,sigma) + '/gt_img' + '/sample_%04d' % num
        path_to_input = output_directory + '/' + args.model_name + '/model_%s_%f' % (mode,sigma) + '/input_img' + '/sample_%04d' % num
        path_to_warped = output_directory + '/' + args.model_name + '/model_%s_%f' % (mode, sigma) + '/warped_img' + '/sample_%04d' % num

        isExists1 = os.path.exists(path_to_disp_est)
        isExists2 = os.path.exists(path_to_img_est)
        isExists3 = os.path.exists(path_to_gt)
        isExists4 = os.path.exists(path_to_input)
        isExists5 = os.path.exists(path_to_warped)

        if not isExists1:
            os.makedirs(path_to_disp_est)
        if not isExists2:
            os.makedirs(path_to_img_est)
        if not isExists3:
            os.makedirs(path_to_gt)
        if not isExists4:
            os.makedirs(path_to_input)
        if not isExists5:
            os.makedirs(path_to_warped)

        SSIM_val_rgb = []
        PSNR_val_rgb = []
        MSE_val_rgb = []
        NRMSE_val_rgb = []

        SSIM_val_y = []
        PSNR_val_y = []
        MSE_val_y = []
        NRMSE_val_y = []

        for i in range(49):
            path_store_img_est = path_to_img_est + '/%03d.png' % i
            path_store_disp_est = path_to_disp_est + '/%03d.png' % i
            path_store_gt = path_to_gt + '/%03d.png' % i
            path_store_warped = path_to_warped + '/%03d.png' % i
            if i == 0:
                disp_est, img_est, gt, input_img, warped_img = sess.run([model.disp, model.color_output, model.label_image, model.input_image,model.wraped_image])
                for j in range(9):
                    imginput = Image.fromarray(np.clip(input_img[0,:,:,j*3:(j+1)*3] * 255, 0, 255).astype(np.uint8))
                    path_store_input = path_to_input + '/%03d.png' % j
                    imginput.save(path_store_input)
            else:
                disp_est, img_est, gt, warped_img = sess.run([model.disp, model.color_output, model.label_image,model.wraped_image])
            imgest = Image.fromarray(np.clip(img_est[0] * 255, 0, 255).astype(np.uint8))
            imggt = Image.fromarray(np.clip(gt[0] * 255, 0, 255).astype(np.uint8))
            imgdispest = Image.fromarray(np.clip((disp_est[0, :, :, 0] + 4) / 8 * 255, 0, 255).astype(np.uint8), mode='L')
            imgwarped = Image.fromarray(np.clip(warped_img[0][:,:,12:15] * 255, 0, 255).astype(np.uint8))

            imgest.save(path_store_img_est)
            imggt.save(path_store_gt)
            imgdispest.save(path_store_disp_est)
            imgwarped.save(path_store_warped)
            if i not in [0, 3, 6, 21, 24, 27, 42, 45, 48]:

                # RGB
                SSIM_val_rgb.append(measure.compare_ssim(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255),
                                                         data_range=255, multichannel=True, gaussian_weights=True))
                PSNR_val_rgb.append(measure.compare_psnr(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255),
                                                         data_range=255))
                MSE_val_rgb.append(measure.compare_mse(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255)))
                NRMSE_val_rgb.append(measure.compare_nrmse(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255)))

                # Y
                SSIM_val_y.append(measure.compare_ssim(color.rgb2ycbcr(np.clip(gt[0][10:-10, 10:-10, :]*255, 0, 255).astype(np.uint8))[:,:,0], color.rgb2ycbcr(np.clip(img_est[0][10:-10, 10:-10, :]*255, 0, 255).astype(np.uint8))[:,:,0], data_range=255))
                PSNR_val_y.append(measure.compare_psnr(color.rgb2ycbcr(np.clip(gt[0][10:-10, 10:-10, :]*255, 0, 255).astype(np.uint8))[:,:,0], color.rgb2ycbcr(np.clip(img_est[0][10:-10, 10:-10, :]*255, 0, 255).astype(np.uint8))[:,:,0], data_range=255))
                MSE_val_y.append(measure.compare_mse(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255)))
                NRMSE_val_y.append(measure.compare_nrmse(np.clip(gt[0][10:-10, 10:-10, :] * 255, 0, 255), np.clip(img_est[0][10:-10, 10:-10, :] * 255, 0, 255)))
        assert len(SSIM_val_rgb) == len(PSNR_val_rgb) == len(MSE_val_rgb) == len(NRMSE_val_rgb) == 40
        assert len(SSIM_val_y) == len(PSNR_val_y) == len(MSE_val_y) == len(NRMSE_val_y) == 40
        ssim_mean_rgb = np.mean(SSIM_val_rgb)
        psnr_mean_rgb = np.mean(PSNR_val_rgb)
        mse_mean_rgb = np.mean(MSE_val_rgb)
        nrmse_mean_rgb = np.mean(NRMSE_val_rgb)
        with open(output_directory + '/' + args.model_name + '/psnr_ssim_rgb.txt', 'a') as fileobject:
            if num == 0:
                title = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('{}_'.format(mode) + str(sigma), 'SSIM', 'PSNR', 'MSE', 'NRMSE')
                fileobject.write(title)
            lines = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('Sample' + str(num) + ':', ssim_mean_rgb, psnr_mean_rgb,
                                                              mse_mean_rgb, nrmse_mean_rgb)
            fileobject.write(lines)
        ssim_mean_y = np.mean(SSIM_val_y)
        psnr_mean_y = np.mean(PSNR_val_y)
        mse_mean_y = np.mean(MSE_val_y)
        nrmse_mean_y = np.mean(NRMSE_val_y)
        with open(output_directory + '/' + args.model_name + '/psnr_ssim_y.txt', 'a') as fileobject:
            if num == 0:
                title = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('{}_'.format(mode) + str(sigma), 'SSIM', 'PSNR', 'MSE', 'NRMSE')
                fileobject.write(title)
            lines = '{:<20}{:<30}{:<30}{:<30}{:<30}\n'.format('Sample' + str(num) + ':', ssim_mean_y, psnr_mean_y,
                                                              mse_mean_y, nrmse_mean_y)
            fileobject.write(lines)

    print('calcul_done.')


def main(_):
    params = parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        num_gpus=args.num_gpus)
    step_list = [110000]
    gauss_list = [0,5,10]
    s_and_p_list = [0.005,0.01]

    for i in step_list:
        for sigma in gauss_list:
            print(i, sigma, 'gauss')
            test(params, i, sigma, mode='gauss')
    for i in step_list:
        for sigma in s_and_p_list:
            print(i,sigma,'s&p')
            test(params, i, sigma, mode='s&p')

if __name__ == '__main__':
    tf.app.run()





