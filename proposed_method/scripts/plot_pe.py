import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from matplotlib import rc
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import gca

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

"""
pe = torch.zeros(max_len, d_model)

position = torch.arange(0, max_len).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (np.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
print(f"Pe shape: {pe.shape}")

pe = pe[:512, :].numpy()
"""
"""
font_manager.findSystemFonts(fontpaths = "/usr/share/fonts/", fontext = "ttf")


font_path = "/usr/share/fonts/truetype/times/Times New Roman.ttf"
props = FontProperties(font_path)

plt.rcParams["font.family"] = "Times"

print(dir(props))
"""
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

#print(matplotlib.font_manager.fontManager.ttflist
