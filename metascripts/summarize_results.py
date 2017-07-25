import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict
import re

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '..', 'logs')

CONFIG_PARAM_NAMES = [
    'randomize_start_pos',
    'translation',
    'rotation',
    'rotation',
    'scale',
    'flashing',
    'num_digits',
    'image',
    'background',
]

def events_to_val_table(event_file_path):
    '''
    Return a NumPy array with step, L_val, PSNR-AUC, and SSIM-AUC along the columns
    :param event_file_path: Path to a tfevents file
    :return:
    '''
    validation_vals = []
    try:
        for event in tf.train.summary_iterator(event_file_path):
            step = event.step
            L_val = None
            psnr_auc = None
            ssim_auc = None
            for v in event.summary.value:
                if v.tag == 'L_val':
                    L_val = v.simple_value
                elif v.tag == 'psnr_auc':
                    psnr_auc = v.simple_value
                elif v.tag == 'ssim_auc':
                    ssim_auc = v.simple_value
            if L_val is not None:
                validation_vals.append(np.array((step, L_val, psnr_auc, ssim_auc)))
    except tf.errors.DataLossError:
        # Ignore incomplete event messages
        pass

    return None if len(validation_vals) == 0 else np.stack(validation_vals, axis=0)


def summarize_val_table(val_tab):
    '''
    Summarize an array of validation values
    :return:
    '''
    num_iters = val_tab[-1, 0]
    L_val_last = val_tab[-1, 1]
    psnr_auc_last = val_tab[-1, 2]
    ssim_auc_last = val_tab[-1, 3]
    L_val_best = np.min(val_tab[:, 1])
    psnr_auc_best = np.max(val_tab[:, 2])
    ssim_auc_best = np.max(val_tab[:, 3])

    x =  OrderedDict([
        ('num_iters', num_iters),
        ('L_val_last', L_val_last),
        ('psnr_auc_last', psnr_auc_last),
        ('ssim_auc_last', ssim_auc_last),
        ('L_val_best', L_val_best),
        ('psnr_auc_best', psnr_auc_best),
        ('ssim_auc_best', ssim_auc_best)
    ])
    return x


def generate_exp_row(exp_name):
    exp_config = {}
    for setting in exp_name.split('+'):
        key, value = tuple(setting.split('='))
        exp_config[key] = value

    # Get summary values
    event_file_list = filter(lambda x: 'events' in x, os.listdir(os.path.join(LOG_DIR, exp_name)))
    # Quit if no event files were found
    if len(event_file_list) == 0:
        return None

    # Get events from latest event file
    last_event_file_path = os.path.join(LOG_DIR, exp_name, event_file_list[-1])
    val_table = events_to_val_table(last_event_file_path)
    # Quit if no validation events occurred
    if val_table is None:
        return None
    summary = summarize_val_table(val_table)

    # Print row
    ret_list = [exp_name]
    for param_name in CONFIG_PARAM_NAMES:
        ret_list.append(exp_config.get(param_name, ''))
    ret_list += [str(x) for x in summary.values()]
    return ','.join(ret_list)


def main():
    col_names = ['exp_name'] \
                + CONFIG_PARAM_NAMES \
                + ['num_iters', 'L_val_last', 'psnr_auc_last', 'ssim_auc_last'] \
                + ['L_val_best', 'psnr_auc_best', 'ssim_auc_best']
    header = ','.join(col_names)

    with open('test.csv', 'w') as f:
        f.write('%s\n' % header)
        # Write the row for each experiment
        for exp_name in os.listdir(LOG_DIR):
            # Only evaluate newer diagonal experiments
            if re.match('|'.join(CONFIG_PARAM_NAMES), exp_name):
                exp_row = generate_exp_row(exp_name)
                if exp_row is not None:
                    f.write('%s\n' % exp_row)

if __name__ == '__main__':
    main()