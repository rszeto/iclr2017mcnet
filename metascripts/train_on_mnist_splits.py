'''
Train multiple MCNets on multiple slices of Moving MNIST. This yields trained models that can be
run on corresponding Moving MNIST validation slices (diagonal experiments) or on other validation
slices (for generalization experiments).
'''

import os
import sys
import glob
from pprint import pprint
import re
from filelock import FileLock
import time
from multiprocessing import Pool
import numpy as np
import traceback
import subprocess
import argparse
from functools import partial


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'MNIST'))
TRAIN_TORONTO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src', 'train_toronto.py'))


def launch_job(t, num_gpus):
    i, cmd = t
    gpu_id = i % num_gpus

    with FileLock('/tmp/gpu_%d.lck' % gpu_id):

        # Test dummy "process"
        # try:
        #     print('%d %s' % (gpu_id, cmd))
        #     np.random.seed(i)
        #     time.sleep(np.random.randint(3))
        # except KeyboardInterrupt:
        #     raise

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            subprocess.check_call(cmd, shell=True, env=env)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            traceback.print_exc()
            # Log failed command
            with FileLock('failed_cmds_log.lck'):
                with open('failed_cmds.log', 'a') as f:
                    f.write(cmd + '\n')


def main(num_gpus, slice_names_file=None):
    if slice_names_file is None:
        video_file_paths = [path for path in glob.glob(MNIST_DATA_DIR + '/*_videos.npy') if '_val_' not in path]
    else:
        with open(slice_names_file, 'r') as f:
            slice_names = [line.strip() for line in f.readlines()]
        video_file_paths = [os.path.join(MNIST_DATA_DIR, '%s_videos.npy' % slice_name) for slice_name in slice_names]

    cmd_fmt = 'python %s --dataset_label=%%s --K=5 --T=5' % TRAIN_TORONTO_PATH
    dataset_labels = [re.search('.*/(.*)_videos\.npy', path).group(1) for path in video_file_paths]
    cmds = [cmd_fmt % dataset_label for dataset_label in dataset_labels][::-1]

    # Start the jobs (let up to four jobs wait for one GPU)
    pool = Pool(4 * num_gpus)
    fn = partial(launch_job, num_gpus=num_gpus)
    res = pool.map_async(fn, enumerate(cmds))

    try:
        # Set timeout to avoid hanging on interrupt
        res.get(9999999)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, help='Number of GPUs on this machine')
    parser.add_argument('--slice_names_file', type=str, help='File path to list of MNIST slice names')
    args = parser.parse_args()
    main(**vars(args))