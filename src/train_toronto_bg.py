import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, sample_freq, val_freq,
         margin):
  # Load video tensor (T x V x H x W)
  video_tensor_path = '../data/MNIST/toronto_bg.npy'
  video_tensor = np.load(video_tensor_path, mmap_mode='r')
  # Load validation set tensor
  video_tensor_path = '../data/MNIST/toronto_bg_val.npy'
  video_tensor_val = np.load(video_tensor_path, mmap_mode='r')
  margin = 0.3
  updateD = True
  updateG = True
  prefix  = ("toronto_bg_MCNET"
          + "_image_size="+str(image_size)
          + "_K="+str(K)
          + "_T="+str(T)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../models/"+prefix+"/"
  samples_dir = "../samples/"+prefix+"/"
  summary_dir = "../logs/"+prefix+"/"

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  if not exists(samples_dir):
    makedirs(samples_dir)
  if not exists(summary_dir):
    makedirs(summary_dir)

  with tf.device("/gpu:%d"%gpu[0]):
    model = MCNET(image_size=[image_size,image_size], c_dim=1,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir)
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars
    )
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha*model.L_img+beta*model.L_GAN, var_list=model.g_vars
    )
    L = alpha*model.L_img+beta*model.L_GAN
    L_sum = tf.summary.scalar("L", L)

    # Add discriminator/generator graph summary nodes
    updateD_tf = tf.placeholder(tf.float32)
    updateD_sum = tf.summary.scalar("updateD", updateD_tf)
    updateG_tf = tf.placeholder(tf.float32)
    updateG_sum = tf.summary.scalar("updateG", updateG_tf)

    # Add validation summary nodes
    L_val = tf.placeholder(tf.float32)
    L_val_sum = tf.summary.scalar("L_val", L_val)
    samples_val = tf.placeholder(tf.float32)
    samples_val_sum = tf.summary.image("samples_val", samples_val, max_outputs=10)

  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                  log_device_placement=False,
                  gpu_options=gpu_options)) as sess:

    tf.global_variables_initializer().run()

    load_result = model.load(sess, checkpoint_dir)
    if load_result:
      print(" [*] Load SUCCESS")
      iters, _ = load_result
    else:
      print(" [!] Load failed...")
      iters = 0

    g_sum = tf.summary.merge([model.L_p_sum,
                              model.L_gdl_sum, model.loss_sum,
                              model.L_GAN_sum, L_sum])
    d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                              model.d_loss_fake_sum, L_sum])
    val_sum = tf.summary.merge([L_val_sum, samples_val_sum])
    update_sum = tf.summary.merge([updateD_sum, updateG_sum])

    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    counter = iters+1
    start_time = time.time()

    with Parallel(n_jobs=batch_size) as parallel:
      while iters < num_iter:
        mini_batches = get_minibatches_idx(video_tensor.shape[1], batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch  = np.zeros((batch_size, image_size, image_size,
                                   K+T, 1), dtype="float32")
            diff_batch = np.zeros((batch_size, image_size, image_size,
                                   K-1, 1), dtype="float32")
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
            output = parallel(delayed(load_moving_mnist_data)(video_tensor, video_index, img_sze, k, t)
                                                 for video_index,img_sze,k,t in zip(batchidx, shapes, Ks, Ts))
            for i in xrange(batch_size):
              seq_batch[i] = output[i][0]
              diff_batch[i] = output[i][1]

            if updateD:
              _, summary_str = sess.run([d_optim, d_sum],
                                         feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                         feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            # Write discriminator/generator update graph
            summary_str = sess.run(update_sum,
                                   feed_dict={updateD_tf: updateD,
                                              updateG_tf: updateG})
            writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errD_real = model.d_loss_real.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errG = model.L_GAN.eval({model.diff_in: diff_batch,
                                     model.xt: seq_batch[:,:,:,K-1],
                                     model.target: seq_batch})
            err = L.eval({model.diff_in: diff_batch,
                          model.xt: seq_batch[:,:,:,K-1],
                          model.target: seq_batch})
  
            print(
                "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, L: %.8f"
                % (iters, time.time() - start_time, errD_fake+errD_real, errG, err)
            )

            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1.-margin) or errD_real > (1.-margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True

            counter += 1

            if np.mod(counter, sample_freq) == 1:
              samples = sess.run([model.G],
                                  feed_dict={model.diff_in: diff_batch,
                                             model.xt: seq_batch[:,:,:,K-1],
                                             model.target: seq_batch})[0]
              samples = samples[0].swapaxes(0,2).swapaxes(1,2)
              sbatch  = seq_batch[0,:,:,K:].swapaxes(0,2).swapaxes(1,2)
              samples = np.concatenate((samples,sbatch), axis=0)
              print("Saving sample ...")
              save_images(samples[:,:,:,::-1], [2, T],
                          samples_dir+"train_%s.png" % (iters))

            if np.mod(counter, val_freq) == 1:
              print("Evaluating model on validation set...")
              mini_batches_val = get_minibatches_idx(video_tensor_val.shape[1], batch_size)
              batch_samples_list = []
              batch_L_list = []
              batch_targets_list = []
              # Forward pass over the whole validation set
              for _, batchidx_val in mini_batches_val:
                if len(batchidx_val) == batch_size:
                  seq_batch_val  = np.zeros((batch_size, image_size, image_size,
                                             K+T, 1), dtype="float32")
                  diff_batch_val = np.zeros((batch_size, image_size, image_size,
                                             K-1, 1), dtype="float32")
                  output_val = parallel(delayed(load_moving_mnist_data)(video_tensor_val, video_index, image_size, K, T)
                                        for video_index in batchidx_val)

                  for i in xrange(batch_size):
                    seq_batch_val[i] = output_val[i][0]
                    diff_batch_val[i] = output_val[i][1]

                  batch_samples, batch_L = sess.run([model.G, L],
                                                      feed_dict={model.diff_in: diff_batch_val,
                                                                 model.xt: seq_batch_val[:,:,:,K-1],
                                                                 model.target: seq_batch_val})
                  batch_samples_list.append(batch_samples)
                  batch_L_list.append(batch_L)
                  batch_targets_list.append(seq_batch_val[:,:,:,K:])

              L_val_np = np.mean(batch_L_list)
              # Stitch targets and predicted frames into an image
              # B x H x W x T x C
              samples_val_np = np.concatenate(batch_samples_list, axis=0)
              targets_val = np.concatenate(batch_targets_list, axis=0)
              batch_images = np.concatenate([samples_val_np, targets_val], axis=3)
              batch_images_list = [merge(x.transpose((2, 0, 1, 3)), [2, T]) for x in batch_images]
              batch_images = np.stack(batch_images_list)

              # Write to TensorBoard log
              summary_str = sess.run(val_sum, feed_dict={L_val: L_val_np,
                                                         samples_val: batch_images})
              writer.add_summary(summary_str, counter-1)

            if np.mod(counter, 500) == 2:
              print("Saving snapshot ...")
              model.save(sess, checkpoint_dir, counter-1)
  
            iters += 1

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=8, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.02, help="GAN loss weight")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=64, help="Image size")
  parser.add_argument("--K", type=int, dest="K",
                      default=10, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=10, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                      help="GPU device id")
  parser.add_argument("--sample_freq", type=int, dest="sample_freq",
                      default=100, help="Number of iterations before saving a sample")
  parser.add_argument("--val_freq", type=int, dest="val_freq",
                      default=100, help="Number of iterations before evaluating on validation set")
  parser.add_argument("--margin", type=float, dest="margin",
                      default=0.3, help="Error margin for updating discriminator/generator")

  args = parser.parse_args()
  main(**vars(args))
