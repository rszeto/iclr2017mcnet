'''
From a list of pairs that refer to a trained model and a dataset, evaluate the model on the dataset.
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
TEST_TORONTO_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src', 'test_toronto.py'))

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


def main(num_gpus, data_model_pairs_file):

    with open(data_model_pairs_file, 'r') as f:
        pairs_str = [line.strip() for line in f.readlines()]
    data_model_pairs = [(x.split()[0], x.split()[1]) for x in pairs_str if len(x) > 0]

    cmd_fmt = 'python %s --prefix=%%s --dataset_label=%%s --K=5 --T=5' % TEST_TORONTO_PATH
    cmds = [cmd_fmt % pair for pair in data_model_pairs]

    # Start the jobs (let up to four jobs wait for one GPU)
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
    parser.add_argument('num_gpus', type=int, help='Number of GPUs on this machine')
    parser.add_argument('--pairs_file', type=str, dest='data_model_pairs_file',
                        default=os.path.join(SCRIPT_DIR, 'data_model_pairs.txt'),
                        help='File path to list of data-model pairs to evaluate')
    args = parser.parse_args()
    main(**vars(args))