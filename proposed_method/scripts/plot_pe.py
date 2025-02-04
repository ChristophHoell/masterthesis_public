"""
    Script File
    Creates a plot of the Sinusoidal Positional Encoding
    Used to create a figure for the thesis
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

max_len = 5000
d_model = 256

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})

pe = getPositionEncoding(seq_len = 256, d = 512, n = 10000).T
ax = plt.subplot()
im = ax.imshow(pe, cmap = "seismic", origin = "lower")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad = 0.05)
cb = plt.colorbar(im, cax = cax)

plt.savefig(os.path.join("/mnt", "e", "TMP", "test", "pe.png"), dpi = 360)
cb.remove()

