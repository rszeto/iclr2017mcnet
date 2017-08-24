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

import skimage.measure as measure
import ssim
from PIL import Image

import threading


def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, sample_freq, val_freq,
         margin, always_update_dg,
         lp_p, gdl_a, target_scale,
         dataset_label):
  # Load video tensor (T x V x H x W)
  video_tensor_path = '../data/MNIST/%s_videos.npy' % dataset_label
  video_tensor = np.load(video_tensor_path)
  seq, diff = get_moving_mnist_tensors(video_tensor, image_size, K, T, target_scale)
  # Load validation set tensor
  video_tensor_path = '../data/MNIST/%s_val_videos.npy' % dataset_label
  video_tensor_val = np.load(video_tensor_path)
  seq_val, diff_val = get_moving_mnist_tensors(video_tensor_val, image_size, K, T, target_scale)

  # Check that number of videos is divisible by batch size
  assert(seq.shape[1] % batch_size == 0)
  assert(seq_val.shape[1] % batch_size == 0)

  # Data queues
  blah_seq_input = tf.placeholder(tf.float32, shape=seq.shape[1:])
  blah_diff_input = tf.placeholder(tf.float32, shape=diff.shape[1:])
  blah_queue = tf.FIFOQueue(seq.shape[0], [tf.float32, tf.float32], shapes=[seq.shape[1:], diff.shape[1:]])
  enqueue_blah_op = blah_queue.enqueue([blah_seq_input, blah_diff_input])
  seq_input, diff_input = blah_queue.dequeue()
  blah_queue_size = blah_queue.size()
  # seq_batch, diff_batch = tf.train.shuffle_batch(dequeue_blah, batch_size=batch_size, capacity=10*batch_size, min_after_dequeue=2*batch_size)

  train_queue = tf.RandomShuffleQueue(10 * batch_size,
                                      2 * batch_size,
                                      [tf.float32, tf.float32],
                                      shapes=[seq.shape[1:], diff.shape[1:]])
  enqueue_train_op = train_queue.enqueue([seq_input, diff_input])
  # seq_batch, diff_batch = train_queue.dequeue_many(batch_size)
  dequeue_train_op = train_queue.dequeue_many(batch_size)
  # enqueue_train_op = train_queue.enqueue([seq_single_input, diff_single_input])

  # # Data loading
  # train_coord = tf.train.Coordinator()
  # # blah_qr = tf.train.QueueRunner(blah_queue, [enqueue_blah_op] * 2)
  # train_qr = tf.train.QueueRunner(train_queue, [enqueue_train_op] * 2)
  # with tf.Session() as sess:
  #   def load_blah_queue():
  #     while True:
  #       for i in xrange(seq.shape[0]):
  #         sess.run(enqueue_blah_op, feed_dict={blah_seq_input: seq[i], blah_diff_input: diff[i]})
  #
  #   t = threading.Thread(target=load_blah_queue)
  #   t.start()
  #
  #   # blah_threads = blah_qr.create_threads(sess, coord=train_coord, start=True)
  #   enqueue_train_threads = train_qr.create_threads(sess, coord=train_coord, start=True)
  #
  #   for x in xrange(100):
  #     print('===')
  #     print(sess.run(blah_queue_size))
  #     a, b = sess.run(dequeue_train_op)
  #     print(a.shape)
  #     print(b.shape)
  #     print(sess.run(blah_queue_size))
  #
  #   train_coord.request_stop()
  #   # train_coord.join(blah_threads)
  #   train_coord.join(enqueue_train_threads)
  #
  # return

  val_queue = tf.FIFOQueue(seq_val.shape[0],
                           [tf.float32, tf.float32],
                           shapes=[seq_val.shape[1:], diff_val.shape[1:]])
  enqueue_val_op = val_queue.enqueue_many([seq_val, diff_val])
  dequeue_val_op = val_queue.dequeue_many(batch_size)
  val_queue_size = val_queue.size()

  updateD = True
  updateG = True
  prefix = dataset_label
  params_arr = [
    "dataset="+dataset_label,
    "image_size="+str(image_size),
    "K="+str(K),
    "T="+str(T),
    "batch_size="+str(batch_size),
    "alpha="+str(alpha),
    "beta="+str(beta),
    "lr="+str(lr),
    "lp_p="+str(lp_p),
    "gdl_a="+str(gdl_a),
    "margin="+str(margin),
    "always_update_dg="+str(always_update_dg),
    "target_scale="+str(target_scale),
    ]

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

  with tf.device("/cpu:0"):
    params = tf.convert_to_tensor(params_arr)
    params_sum = tf.summary.text("params", params)

  with tf.device("/gpu:%d"%gpu[0]):
    model = MCNET(image_size=[image_size,image_size], c_dim=1,
                  K=K, batch_size=batch_size, T=T,
                  p=lp_p, alpha=gdl_a,
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
    psnr_auc = tf.placeholder(tf.float32)
    psnr_auc_sum = tf.summary.scalar("psnr_auc", psnr_auc)
    psnr_plot = tf.placeholder(tf.uint8)
    psnr_plot_sum = tf.summary.image("psnr_val", psnr_plot, max_outputs=1)
    ssim_auc = tf.placeholder(tf.float32)
    ssim_auc_sum = tf.summary.scalar("ssim_auc", ssim_auc)
    ssim_plot = tf.placeholder(tf.uint8)
    ssim_plot_sum = tf.summary.image("ssim_val", ssim_plot, max_outputs=1)

  gpu_options = tf.GPUOptions(allow_growth=True)

  # Data loading
  train_coord = tf.train.Coordinator()
  train_qr = tf.train.QueueRunner(train_queue, [enqueue_train_op] * 2)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    def load_blah_queue():
      while True:
        for i in xrange(seq.shape[0]):
          sess.run(enqueue_blah_op, feed_dict={blah_seq_input: seq[i], blah_diff_input: diff[i]})

    t = threading.Thread(target=load_blah_queue)
    t.start()

    tf.global_variables_initializer().run()
    enqueue_train_threads = train_qr.create_threads(sess, coord=train_coord, start=True)

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
    val_sum = tf.summary.merge([L_val_sum, samples_val_sum, psnr_plot_sum, psnr_auc_sum, ssim_plot_sum, ssim_auc_sum])
    update_sum = tf.summary.merge([updateD_sum, updateG_sum])

    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Print parameters to TensorBoard
    params_sum_str = sess.run(params_sum)
    writer.add_summary(params_sum_str, 0)

    counter = iters+1
    start_time = time.time()

    with Parallel(n_jobs=batch_size) as parallel:
      while iters < num_iter:
        mini_batches = get_minibatches_idx(video_tensor.shape[1], batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            # seq_batch  = np.zeros((batch_size, image_size, image_size,
            #                        K+T, 1), dtype="float32")
            # diff_batch = np.zeros((batch_size, image_size, image_size,
            #                        K-1, 1), dtype="float32")
            # Ts = np.repeat(np.array([T]),batch_size,axis=0)
            # Ks = np.repeat(np.array([K]),batch_size,axis=0)
            # shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
            # output = parallel(delayed(load_moving_mnist_data)(video_tensor, video_index, img_sze, k, t, target_scale)
            #                                      for video_index,img_sze,k,t in zip(batchidx, shapes, Ks, Ts))
            # for i in xrange(batch_size):
            #   seq_batch[i] = output[i][0]
            #   diff_batch[i] = output[i][1]

            # Get a training batch
            seq_batch, diff_batch = sess.run(dequeue_train_op)
            input_dict = {
              model.diff_in: diff_batch,
              model.xt: seq_batch[:,:,:,K-1],
              model.target: seq_batch
            }

            if updateD:
              _, summary_str = sess.run([d_optim, g_sum], feed_dict=input_dict)
              # _, summary_str = sess.run([d_optim, d_sum])
              writer.add_summary(summary_str, counter)

            if updateG:
              _, summary_str = sess.run([g_optim, g_sum], feed_dict=input_dict)
              # _, summary_str = sess.run([g_optim, g_sum])
              writer.add_summary(summary_str, counter)

            # Write discriminator/generator update graph
            summary_str = sess.run(update_sum,
                                   feed_dict={updateD_tf: updateD,
                                              updateG_tf: updateG})
            writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval(input_dict)
            errD_real = model.d_loss_real.eval(input_dict)
            errG = model.L_GAN.eval(input_dict)
            err = L.eval(input_dict)

            print(
              "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, L: %.8f, updateD: %r, updateG: %r"
              % (iters, time.time() - start_time, errD_fake+errD_real, errG, err, updateD, updateG)
            )

            if not always_update_dg:
              if errD_fake < margin or errD_real < margin:
                updateD = False
              if errD_fake > (1.-margin) or errD_real > (1.-margin):
                updateG = False
              if not updateD and not updateG:
                updateD = True
                updateG = True

            counter += 1

            if np.mod(counter, sample_freq) == 1:
              samples = sess.run([model.G], feed_dict=input_dict)[0]
              samples = samples[0].swapaxes(0,2).swapaxes(1,2)
              sbatch  = seq_batch[0,:,:,K:].swapaxes(0,2).swapaxes(1,2)
              samples = np.concatenate((samples,sbatch), axis=0)
              print("Saving sample ...")
              save_images(samples[:,:,:,::-1], [2, T],
                          samples_dir+"train_%s.png" % (iters),
                          target_scale)

            if np.mod(counter, val_freq) == 1:
              print("Evaluating model on validation set...")
              batch_samples_list = []
              batch_L_list = []
              batch_targets_list = []
              mini_batches_val = get_minibatches_idx(video_tensor_val.shape[1], batch_size)

              # Enqueue all validation videos
              sess.run(enqueue_val_op)
              # Forward pass over the whole validation set
              for batch_num, batchidx_val in mini_batches_val:
                if len(batchidx_val) == batch_size:
                  seq_batch_val, diff_batch_val = sess.run(dequeue_val_op)

                  batch_samples, batch_L = sess.run([model.G, L],
                                                    feed_dict={model.diff_in: diff_batch_val,
                                                               model.xt: seq_batch_val[:,:,:,K-1],
                                                               model.target: seq_batch_val})
                  batch_samples_list.append(batch_samples)
                  batch_L_list.append(batch_L)
                  batch_targets_list.append(seq_batch_val[:,:,:,K:])

              # Check that the validation queue was emptied
              assert(sess.run(val_queue_size) == 0)

              L_val_np = np.mean(batch_L_list)
              # B x H x W x T x C
              samples_val_np = inverse_transform(np.concatenate(batch_samples_list, axis=0), target_scale)
              targets_val = inverse_transform(np.concatenate(batch_targets_list, axis=0), target_scale)

              # Collect per-frame PSNR and SSIM
              psnr_values = np.zeros((video_tensor_val.shape[1], T))
              ssim_values = np.zeros((video_tensor_val.shape[1], T))
              for video_idx in xrange(samples_val_np.shape[0]):
                for frame_idx in xrange(T):
                  sample_frame = np.array(255 * samples_val_np[video_idx, :, :, frame_idx, :], dtype=np.uint8)
                  target_frame = np.array(255 * targets_val[video_idx, :, :, frame_idx, :], dtype=np.uint8)
                  # Clip sample values outside of [0, 255]
                  sample_frame = np.minimum(np.maximum(sample_frame, 0), 255)
                  psnr_values[video_idx, frame_idx] = measure.compare_psnr(sample_frame, target_frame)
                  ssim_values[video_idx, frame_idx] = ssim.compute_ssim(
                    Image.fromarray(cv2.cvtColor(target_frame, cv2.COLOR_GRAY2BGR)),
                    Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_GRAY2BGR))
                  )

              # Generate plots and AUC of PSNR and SSIM averaged over all videos
              psnr_curve = psnr_values.mean(axis=0)
              ssim_curve = ssim_values.mean(axis=0)
              psnr_auc_np = np.sum(psnr_curve)
              ssim_auc_np = np.sum(ssim_curve)
              psnr_plot_np = plot_to_image(range(1, T+1), psnr_curve, [1, T+1, 0, 50])
              ssim_plot_np = plot_to_image(range(1, T+1), ssim_curve, [1, T+1, 0, 1])

              # Stitch targets and predicted frames into an image
              batch_images = np.concatenate([samples_val_np, targets_val], axis=3)
              batch_images_list = [merge(x.transpose((2, 0, 1, 3)), [2, T]) for x in batch_images]
              batch_images = np.stack(batch_images_list)
              # Clip values outside of [0, 1], then scale to [0, 255]
              batch_images = 255 * np.minimum(np.maximum(batch_images, 0), 255)
              # Convert to avoid TensorFlow scaling weirdness
              batch_images = batch_images.astype(np.uint8)

              # Write to TensorBoard log
              summary_str = sess.run(val_sum, feed_dict={L_val: L_val_np,
                                                         samples_val: batch_images,
                                                         psnr_plot: psnr_plot_np,
                                                         psnr_auc: psnr_auc_np,
                                                         ssim_plot: ssim_plot_np,
                                                         ssim_auc: ssim_auc_np})
              writer.add_summary(summary_str, counter-1)

            if np.mod(counter, 500) == 2:
              print("Saving snapshot ...")
              model.save(sess, checkpoint_dir, counter-1)

            iters += 1
            if iters >= num_iter:
              break

    train_coord.request_stop()
    train_coord.join(enqueue_train_threads)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=8, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=2e-5, help="GAN loss weight")
  parser.add_argument("--image_size", type=int, dest="image_size",
                      default=64, help="Image size")
  parser.add_argument("--K", type=int, dest="K",
                      default=10, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=10, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default=[0],
                      help="GPU device id")
  parser.add_argument("--sample_freq", type=int, dest="sample_freq",
                      default=100, help="Number of iterations before saving a sample")
  parser.add_argument("--val_freq", type=int, dest="val_freq",
                      default=500, help="Number of iterations before evaluating on validation set")
  parser.add_argument("--margin", type=float, dest="margin",
                      default=0.3, help="Error margin for updating discriminator/generator")
  parser.add_argument("--always_update_dg", type=bool, dest="always_update_dg",
                      default=False, help="Whether to always update both discriminator and generator")
  parser.add_argument("--lp_p", type=float, dest="lp_p",
                      default=2.0, help="Hyperparameter for L_p loss")
  parser.add_argument("--gdl_a", type=float, dest="gdl_a",
                      default=1.0, help="Hyperparameter for L_gdl loss")
  parser.add_argument("--target_scale", type=float, dest="target_scale",
                      default=0.75, help="How much to scale model targets")
  parser.add_argument("--dataset_label", type=str, dest="dataset_label", required=True,
                      help="Name of the dataset (i.e. X in ../data/MNIST/X_videos.npy)")

  args = parser.parse_args()
  main(**vars(args))
