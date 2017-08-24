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
from filelock import FileLock, Timeout
import time
from multiprocessing import Pool
import numpy as np
import traceback
import subprocess
import argparse
from functools import partial


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'MNIST'))

TRAIN_SCRIPT_PATH_MAP = {
    'mcnet': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src', 'train_toronto.py')),
    'mcnet_content_lstm': os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src', 'train_toronto_mcnet_content_lstm.py'))
}


def launch_job(t, num_gpus):
    i, cmd = t
    gpu_id = 0
    launched_job = False

    while not launched_job:
        # Try to acquire lock for current GPU
        lock = FileLock('/tmp/gpu_%d.lck' % gpu_id, timeout=0)
        try:
            with lock.acquire():
                # # Test dummy "process"
                # try:
                #     print('%d %s' % (gpu_id, cmd))
                #     np.random.seed(i)
                #     time.sleep(np.random.randint(3))
                # except KeyboardInterrupt:
                #     raise
                # finally:
                #     launched_job = True

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
                finally:
                    launched_job = True
        except Timeout:
            # Try the next GPU if current GPU is used
            gpu_id = gpu_id + 1 % num_gpus


def main(arch, num_gpus, slice_names_file):
    if slice_names_file is None:
        video_file_paths = [path for path in glob.glob(MNIST_DATA_DIR + '/*_videos.npy') if '_val_' not in path]
    else:
        with open(slice_names_file, 'r') as f:
            slice_names = [line.strip() for line in f.readlines()]
        # Filter out empty lines or commented lines
        slice_names = filter(lambda x: len(x) > 0 and not x.startswith('#'), slice_names)
        video_file_paths = [os.path.join(MNIST_DATA_DIR, '%s_videos.npy' % slice_name) for slice_name in slice_names]

    train_script_path = TRAIN_SCRIPT_PATH_MAP.get(arch, None)
    if train_script_path is None:
        print('%s is not a valid architecture. Quitting' % arch)
        return

    cmd_fmt = 'python %s --dataset_label=%%s --K=10 --T=5' % train_script_path
    dataset_labels = [re.search('.*/(.*)_videos\.npy', path).group(1) for path in video_file_paths]
    cmds = [cmd_fmt % dataset_label for dataset_label in dataset_labels][::-1]

    # Start the jobs
    pool = Pool(num_gpus)
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
    parser.add_argument('arch', type=str, help='Label of the MCNet architecture to use')
    parser.add_argument('num_gpus', type=int, help='Number of GPUs on this machine')
    parser.add_argument('--slice_names_file', type=str,
                        default=os.path.join(SCRIPT_DIR, 'slice_names.txt'),
                        help='File path to list of MNIST slice names')
    args = parser.parse_args()
    main(**vars(args))
