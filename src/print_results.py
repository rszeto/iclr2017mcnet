import numpy as np
from Tkinter import Tk
from tkFileDialog import askopenfilename
import matplotlib.pyplot as plt
import re

# Select file path (https://stackoverflow.com/a/3579625)
Tk().withdraw()
filename = askopenfilename(initialdir='../results/quantitative')
if not filename: exit()

# Parse the path to get dataset, iters, and batch size
parse = re.search('.*/(KTH|UCF101)/.*(batch_size=[0-9]+).*?([0-9]+).npz', filename)
test_set_label = parse.group(1)
batch_size = parse.group(2)
iter_num = int(parse.group(3))
title = '%s, %s, iters=%d' % (test_set_label, batch_size, iter_num)

results = np.load(filename)
psnr = results['psnr'].mean(axis=0)
ssim = results['ssim'].mean(axis=0)

print('PSNR: %s' % psnr)
print('SSIM: %s' % ssim)

fig = plt.figure(figsize=(4.8, 5.5))
x = np.arange(len(psnr))+1

# Draw PSNR plot
ax1 = plt.subplot(211)
plt.plot(x, psnr)
plt.axis([1, len(psnr), 20, 38])
plt.xticks(x)
plt.yticks(np.arange(20, 39, 2))
ax1.set_ylabel('PSNR')
ax1.grid(True)
for line in ax1.get_xgridlines() + ax1.get_ygridlines():
    line.set_linestyle('dotted')

# Title here so it appears above both plots
plt.title(title)

# Draw SSIM plot
ax2 = plt.subplot(212)
ax2.plot(x, ssim)
plt.axis([1, len(psnr), 0.6, 1.0])
plt.xticks(x)
plt.yticks(np.arange(0.6, 1.01, 0.1))
ax2.set_ylabel('SSIM')
ax2.grid(True)
for line in ax2.get_xgridlines() + ax2.get_ygridlines():
    line.set_linestyle('dotted')

plt.tight_layout()
# Hack around plt.show() because it doesn't return for some reason
plt.draw()
while True:
    w = plt.waitforbuttonpress()