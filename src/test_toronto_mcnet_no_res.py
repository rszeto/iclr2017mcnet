import os
import cv2
import sys
import time
import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from mcnet_no_res import MCNET_NO_RES
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw

def eval(prefix, image_size, K, T, E, gpu, target_scale, dataset_label,
         video_tensor, c_dim, checkpoint_dir, best_model):
  assert(K+T+E <= video_tensor.shape[0])
  with tf.device("/gpu:%d" % gpu[0]):
    # Set up placeholder for discriminator
    disc_data = tf.placeholder(tf.float32, shape=[1, image_size, image_size, K + T])

    model = MCNET_NO_RES(image_size=[image_size, image_size], batch_size=1, K=K,
                  T=T+E, c_dim=c_dim, checkpoint_dir=checkpoint_dir,
                  is_train=False, target_scale=target_scale)

    # Add personal discriminator
    with tf.variable_scope("DIS", reuse=False):
      D__, _ = model.discriminator(disc_data)
    # Recreate the saver so the vars for the personal discriminator are loaded
    model.saver = tf.train.Saver()

  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    tf.global_variables_initializer().run()

    load_result = model.load(sess, checkpoint_dir, best_model)
    if load_result:
      print(" [*] Load SUCCESS")
      _, model_name = load_result
    else:
      print(" [!] Load failed... exiting")
      return

    quant_dir = "../results/quantitative/MNIST/arch=MCNET_NO_RES/data=%s/model=%s/" % (dataset_label, prefix)
    save_path = quant_dir + "results_model=" + model_name + ".npz"
    if not exists(quant_dir):
      makedirs(quant_dir)

    vid_names = []
    psnr_err = np.zeros((0, T+E))
    ssim_err = np.zeros((0, T+E))
    disc_output = np.zeros((video_tensor.shape[1], E))
    for i in xrange(video_tensor.shape[1]):
      print("Video " + str(i) + "/" + str(video_tensor.shape[1]))

      savedir = "../results/images/MNIST/arch=MCNET_NO_RES/data=%s/model=%s/%04d" % (dataset_label, prefix, i)

      seq_batch = transform(
        video_tensor[:, i, :, :, np.newaxis, np.newaxis],
        target_scale=target_scale
      ).transpose((3, 1, 2, 0, 4)).astype("float32")

      diff_batch = np.zeros((1, image_size, image_size,
                             K - 1, 1), dtype="float32")
      for t in xrange(1, K):
        prev = inverse_transform(seq_batch[0, :, :, t - 1], target_scale=target_scale)
        next = inverse_transform(seq_batch[0, :, :, t], target_scale=target_scale)
        diff = next.astype("float32") - prev.astype("float32")
        diff_batch[0, :, :, t - 1] = diff

      true_data = seq_batch[:, :, :, K:K+T+E, :].copy()
      pred_data_shape = list(true_data.shape)
      pred_data = np.zeros(pred_data_shape, dtype="float32")
      xt = seq_batch[:, :, :, K - 1]
      pred_data[0] = sess.run(model.G,
                              feed_dict={model.diff_in: diff_batch,
                                         model.xt: xt})

      if not os.path.exists(savedir):
        os.makedirs(savedir)

      cpsnr = np.zeros((K + T + E,))
      cssim = np.zeros((K + T + E,))
      pred_data = np.concatenate((seq_batch[:, :, :, :K], pred_data), axis=3)
      true_data = np.concatenate((seq_batch[:, :, :, :K], true_data), axis=3)

      for t in xrange(K + T + E):
        pred = inverse_transform(pred_data[0, :, :, t], target_scale=target_scale)
        # Clip pred within [0, 1]
        pred = 255 * np.minimum(np.maximum(pred, 0), 1)
        # Convert pred to uint8
        pred = pred.astype(np.uint8)
        target = (inverse_transform(true_data[0, :, :, t], target_scale=target_scale) * 255).astype("uint8")

        cpsnr[t] = measure.compare_psnr(pred, target)
        cssim[t] = ssim.compute_ssim(Image.fromarray(cv2.cvtColor(target,
                                                                  cv2.COLOR_GRAY2BGR)),
                                     Image.fromarray(cv2.cvtColor(pred,
                                                                  cv2.COLOR_GRAY2BGR)))

        pred = draw_frame(pred, t < K)
        target = draw_frame(target, True)
        both = merge(np.stack([target, pred], axis=0), [1, 2])

        cv2.imwrite(savedir + "/pred_" + "{0:04d}".format(t) + ".png", pred)
        cv2.imwrite(savedir + "/gt_" + "{0:04d}".format(t) + ".png", target)
        cv2.imwrite(savedir + "/both_" + "{0:04d}".format(t) + ".png", both)

        if t > K+T:
          # Save discriminator score
          d_out = sess.run(D__, feed_dict={
            disc_data: inverse_transform(pred_data[:, :, :, t-K-T:t, 0], target_scale=target_scale)
          })
          disc_output[i, t-K-T] = d_out

      cmd1 = "rm " + savedir + "/pred.gif"
      cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
              "/pred_%04d.png " + savedir + "/pred.gif")
      cmd3 = "rm " + savedir + "/pred*.png"

      # Comment out "system(cmd3)" if you want to keep the output images
      # Otherwise only the gifs will be kept
      system(cmd1);
      system(cmd2);
      system(cmd3);

      cmd1 = "rm " + savedir + "/gt.gif"
      cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
              "/gt_%04d.png " + savedir + "/gt.gif")
      cmd3 = "rm " + savedir + "/gt*.png"

      # Comment out "system(cmd3)" if you want to keep the output images
      # Otherwise only the gifs will be kept
      system(cmd1);
      system(cmd2);
      system(cmd3);

      cmd1 = "rm " + os.path.dirname(savedir) + "/both_%04d.gif" % i
      cmd2 = ("ffmpeg -f image2 -framerate 7 -i " + savedir +
              "/both_%04d.png " + os.path.dirname(savedir) + "/both_%04d.gif" % i)
      cmd3 = "rm " + savedir + "/both*.png"

      # Comment out "system(cmd3)" if you want to keep the output images
      # Otherwise only the gifs will be kept
      system(cmd1);
      system(cmd2);
      system(cmd3);

      psnr_err = np.concatenate((psnr_err, cpsnr[None, K:]), axis=0)
      ssim_err = np.concatenate((ssim_err, cssim[None, K:]), axis=0)

    np.savez(save_path, psnr=psnr_err, ssim=ssim_err, disc_output=disc_output)
    print("Results saved to " + save_path)

def main(prefix, image_size, K, T, E, gpu, target_scale, dataset_label):
  test_video_tensor_path = '../data/MNIST/%s_test_videos.npy' % dataset_label
  test_video_tensor = np.load(test_video_tensor_path, mmap_mode='r')
  long_video_tensor_path = '../data/MNIST/%s_long_videos.npy' % dataset_label
  long_video_tensor = np.load(long_video_tensor_path, mmap_mode='r')

  c_dim = 1

  checkpoint_dir = "../models/MCNET_NO_RES/"+prefix+"/"
  best_model = None # will pick last model

  eval(prefix, image_size, K, T, E, gpu, target_scale, dataset_label + '_long',
       long_video_tensor, c_dim, checkpoint_dir, best_model)
  tf.reset_default_graph()
  eval(prefix, image_size, K, T, 0, gpu, target_scale, dataset_label,
       test_video_tensor, c_dim, checkpoint_dir, best_model)
  tf.reset_default_graph()

  print("Done.")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--prefix", type=str, dest="prefix", required=True,
                      help="Prefix for log/snapshot")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=64, help="Image size")
  parser.add_argument("--K", type=int, dest="K",
                      default=10, help="Number of input images")
  parser.add_argument("--T", type=int, dest="T",
                      default=10, help="Number of steps into the future to compare to ground truth")
  parser.add_argument("--E", type=int, dest="E",
                      default=0, help="Number of steps past T to generate frames for")
  parser.add_argument("--target_scale", type=float, dest="target_scale",
                      default=0.75, help="How much to scale model targets")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu",
                      default=[0], help="GPU device id")
  parser.add_argument("--dataset_label", type=str, dest="dataset_label", required=True,
                      help="Name of the dataset (i.e. X in ../data/MNIST/X_val_videos.npy)")

  args = parser.parse_args()
  main(**vars(args))
