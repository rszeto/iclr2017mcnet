import numpy as np
from Tkinter import Tk
from tkFileDialog import askopenfilename
import matplotlib.pyplot as plt
import re
import os


def plot_graphs(result_paths):
    test_set_label = None
    x_range = None
    for filename in result_paths:
        # Parse the path to get dataset
        print(filename)
        parse = re.search('(KTH|UCF101|MNIST.+?)/', filename)
        cur_test_set_label = parse.group(1)
        assert(test_set_label is None or test_set_label == cur_test_set_label)
        test_set_label = cur_test_set_label

        results = np.load(filename)
        psnr = results['psnr']
        x = np.arange(psnr.shape[1])+1
        # Keep the longest x range
        if x_range is None or len(x) > len(x_range):
            x_range = x


    # Draw PSNR plot
    ax1 = plt.subplot(211)
    for filename in result_paths:
        results = np.load(filename)
        psnr = results['psnr'].mean(axis=0)
        psnr_plot = np.inf * np.ones(len(x_range))
        psnr_plot[:len(psnr)] = psnr
        plt.plot(x_range, psnr_plot, label=os.path.basename(filename))
    plt.xticks(x_range)
    ax1.set_ylabel('PSNR')
    ax1.grid(True)
    for line in ax1.get_xgridlines() + ax1.get_ygridlines():
        line.set_linestyle('dotted')
    if test_set_label == 'KTH':
        plt.axis([1, len(psnr), 20, 35])
        plt.yticks(np.arange(20, 35, 2))
    elif test_set_label == 'UCF101':
        plt.axis([1, len(psnr), 10, 33])
        plt.yticks(np.arange(10, 33, 5))
    plt.legend()

    # Draw SSIM plot
    ax2 = plt.subplot(212)
    for filename in result_paths:
        results = np.load(filename)
        ssim = results['ssim'].mean(axis=0)
        ssim_plot = np.inf * np.ones(len(x_range))
        ssim_plot[:len(ssim)] = ssim
        ax2.plot(x_range, ssim_plot, label=os.path.basename(filename))
    plt.xticks(x_range)
    ax2.set_ylabel('SSIM')
    ax2.grid(True)
    for line in ax2.get_xgridlines() + ax2.get_ygridlines():
        line.set_linestyle('dotted')
    if test_set_label == 'KTH':
        plt.axis([1, len(psnr), 0.6, 1.0])
        plt.yticks(np.arange(0.6, 1.01, 0.1))
    elif test_set_label == 'UCF101':
        plt.axis([1, len(psnr), 0.35, 1.0])
        plt.yticks(np.arange(0.4, 1.01, 0.1))
    plt.legend()


def main():
    # Get result paths
    result_paths = []

    while True:
        # Select file path (https://stackoverflow.com/a/3579625)
        Tk().withdraw()
        filename = askopenfilename(initialdir='../results/quantitative')
        if filename:
            result_paths.append(filename)
        else:
            break

    # Quit if no files were given
    if len(result_paths) == 0:
        return

    fig = plt.figure(figsize=(4.8, 5.5))
    plot_graphs(result_paths)
    plt.tight_layout()

    # Hack around plt.show() because it doesn't return for some reason
    plt.draw()
    while True:
        w = plt.waitforbuttonpress()


if __name__ == '__main__':
    main()