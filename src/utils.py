"""
Some codes from https://github.com/Newmu/dcgan_code
"""
# Make plots without X11 server
import matplotlib
matplotlib.use('Agg')

import cv2
import random
import imageio
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
from warnings import warn


def transform(image, target_scale=1.0):
    return target_scale*(image/127.5 - 1.)


def inverse_transform(images, target_scale=1.0):
    return (images/target_scale+1.)/2.


def save_images(images, size, image_path, target_scale):
  return imsave(inverse_transform(images, target_scale)*255., size, image_path)


def merge(images, size):
  '''
  Tile the given batch of images into a grid
  :param images: B x H x W x C NumPy array
  :param size: list with values [H_grid, W_grid]
  :return:
  '''
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  tiled_images = merge(images, size)
  return scipy.misc.toimage(tiled_images, cmin=0, cmax=255).save(path)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 


def load_kth_data(f_name, data_path, image_size, K, T):
  '''
  :param f_name: Line from <data_path>/train_data_list.txt
  :param data_path: Path to video folder
  :param image_size: int64 indicating image height and width
  :param K: Number of past frames
  :param T: Number of future frames
  :return:
    seq: np.float32 array with shape (image_size, image_size, K+T, 1) of video frames. Values are in [-1, 1]
    diff: np.float32 with shape (image_size, image_size, K-1, 1) of frame deltas. Values are in [-1, 1]
  '''
  flip = np.random.binomial(1,.5,1)[0]
  tokens = f_name.split()
  vid_path = data_path + tokens[0] + "_uncomp.avi"
  vid = imageio.get_reader(vid_path,"ffmpeg")
  low = int(tokens[1])
  high = np.min([int(tokens[2]),vid.get_length()])-K-T+1
  if low == high:
    stidx = 0 
  else:
    if low >= high: print(vid_path)
    stidx = np.random.randint(low=low, high=high)
  seq = np.zeros((image_size, image_size, K+T, 1), dtype="float32")
  for t in xrange(K+T):
    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx+t),
                       (image_size,image_size)),
                       cv2.COLOR_RGB2GRAY)
    seq[:,:,t] = transform(img[:,:,None])

  if flip == 1:
    seq = seq[:,::-1]

  diff = np.zeros((image_size, image_size, K-1, 1), dtype="float32")
  for t in xrange(1,K):
    prev = inverse_transform(seq[:,:,t-1])
    next = inverse_transform(seq[:,:,t])
    diff[:,:,t-1] = next.astype("float32")-prev.astype("float32")

  return seq, diff


def load_s1m_data(f_name, data_path, trainlist, K, T):
  flip = np.random.binomial(1,.5,1)[0]
  vid_path = os.path.join(data_path, f_name)
  img_size = [240,320]

  while True:
    try:
      vid = imageio.get_reader(vid_path,"ffmpeg")
      low = 1
      high = vid.get_length()-K-T+1
      if low == high:
        stidx = 0
      else:
        stidx = np.random.randint(low=low, high=high)
      seq = np.zeros((img_size[0], img_size[1], K+T, 3),
                     dtype="float32")
      for t in xrange(K+T):
        img = cv2.resize(vid.get_data(stidx+t),
                         (img_size[1],img_size[0]))[:,:,::-1]
        seq[:,:,t] = transform(img)

      if flip == 1:
        seq = seq[:,::-1]

      diff = np.zeros((img_size[0], img_size[1], K-1, 1),
                      dtype="float32")
      for t in xrange(1,K):
        prev = inverse_transform(seq[:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq[:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff[:,:,t-1,0] = (next.astype("float32")-prev.astype("float32"))/255.
      break
    except Exception:
      warn('Failed to find %s' % vid_path)
      # In case the current video is bad load a random one
      rep_idx = np.random.randint(low=0, high=len(trainlist))
      f_name = trainlist[rep_idx]
      vid_path = os.path.join(data_path, f_name)

  return seq, diff

def load_moving_mnist_data(video_tensor, video_index, image_size, K, T, target_scale):
  if image_size != video_tensor.shape[2] or image_size != video_tensor.shape[3]:
    raise ValueError('Image size %d must match video dimensions %s' % (image_size, video_tensor.shape[2:]))

  seq = transform(video_tensor[:K+T, video_index, :, :], target_scale)
  seq = seq.transpose((1, 2, 0))[:, :, :, np.newaxis]
  diff = np.zeros((image_size, image_size, K - 1, 1), dtype="float32")
  for t in xrange(1,K):
    prev = inverse_transform(seq[:,:,t-1], target_scale)
    next = inverse_transform(seq[:,:,t], target_scale)
    diff[:,:,t-1] = next.astype("float32")-prev.astype("float32")

  return seq, diff


def get_moving_mnist_tensors(video_tensor, image_size, K, T, target_scale):
  if image_size != video_tensor.shape[2] or image_size != video_tensor.shape[3]:
    raise ValueError('Image size %d must match video dimensions %s' % (image_size, video_tensor.shape[2:]))

  seq = transform(video_tensor[:K + T, :, :, :], target_scale)
  seq = seq.transpose((1, 2, 3, 0))
  diff = video_tensor[1:K].astype('float32') - video_tensor[:K-1].astype('float32')
  diff = diff.transpose((1, 2, 3, 0)) / 255.0

  return np.expand_dims(seq, -1), np.expand_dims(diff, -1)

def load_moving_mnist_data_2(seq, diff, video_index):
  return seq[video_index][:, :, :, np.newaxis], diff[video_index][:, :, :, np.newaxis]


def get_normalized_identities(identities, target_scale):
  return np.expand_dims(transform(identities, target_scale), -1)


def plot_to_image(x, y, lims):
  '''
  Plot y vs. x and return the graph as a NumPy array
  :param x: X values
  :param y: Y values
  :param lims: [x_start, x_end, y_start, y_end]
  :return:
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(x, y)
  ax.axis(lims)
  plot_buf = gen_plot(fig)
  im = np.array(Image.open(plot_buf), dtype=np.uint8)
  im = np.expand_dims(im, axis=0)
  plt.close(fig)
  return im


def gen_plot(fig):
  """
  Create a pyplot plot and save to buffer.
  https://stackoverflow.com/a/38676842
  """
  buf = io.BytesIO()
  fig.savefig(buf, format='png')
  buf.seek(0)
  return buf
