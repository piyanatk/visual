import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
# telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
# bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
bandwidth = [0.08]
stat = ['var', 'skew', 'kurt']
nf = 5

l1 = 'r'
l2 = 'b'
s1 = '#fdae61'
s2 = '#abd9e9'
alpha = 0.5
handles = [Line2D([], [], c=l1, ls='-'), Line2D([], [], c=l2, ls='--'),
           Patch(fc=s1, alpha=alpha), Patch(fc=s2, alpha=alpha)]
labels = ['Binning', 'Windowing', 'Binning 20-Field Sample Error',
          'Windowing 20-Field Sample Error']

st = 'kurt'
bw = 1.00
for i in range(int(200 / nf)):  # Pages
    pt = range(nf*i, nf*i+nf)  # Pointing
    plt.close()
    fig, ax = plt.subplots(nrows=len(telescope), ncols=nf, sharex='all',
                           sharey='all', figsize=(11, 8.5))
    for j in range(len(telescope)):  # Rows, telescope
        tl = telescope[j]
        pn1 = pd.read_hdf(
            '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_binning.h5'
            .format(data_dir, tl, tl, bw)
        )
        y1mean = pn1.mean(axis=0)
        y1err = pn1.std(axis=0)
        y1max = y1mean + y1err
        y1min = y1mean - y1err
        pn2 = pd.read_hdf(
            '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_windowing.h5'
            .format(data_dir, tl, tl, bw)
        )
        y2mean = pn2.mean(axis=0)
        y2err = pn2.std(axis=0)
        y2max = y2mean + y2err
        y2min = y2mean - y2err
        x = pn1.major_axis.values
        for k in range(nf):  # Columns, fields
            field = nf * i + k
            y1 = pn1.iloc[field][st].values
            y2 = pn2.iloc[field][st].values
            ax[j, k].fill_between(x, y1min[st].values,
                                  y1max[st].values,
                                  color=s1, alpha=alpha)
            ax[j, k].fill_between(x, y2min[st].values,
                                  y2max[st].values,
                                  color=s2, alpha=alpha)
            ax[j, k].plot(x, y1, c=l1, ls='-')
            ax[j, k].plot(x, y2, c=l2, ls='--')
            ax[j, k].axhline(0, c='k', ls=':')
            if j == 0:
                ax[j, k].set_title('Field {:d}'.format(field))
            if k == nf-1:
                ax[j, k].set_ylabel('{:s}'.format(tl))
                ax[j, k].yaxis.set_label_position('right')
            ax[j, k].set_ylim(-1, 2.5)
    fig.text(0.01, 0.5, 'Kurtosis', rotation='vertical', ha='left', va='center')
    fig.text(0.5, 0.05, 'Observed Frequency [MHz]', ha='center', va='bottom')
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
    fig.tight_layout(rect=[0.01, 0.05, 1, 1], w_pad=0, h_pad=0)
    outdir = '/Users/piyanat/Google/research/hera1p/plots/' \
             'mc_stats_heraxx/bw{:.2f}MHz/{:s}'.format(bw, st)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig('{:s}/mc_stats_heraxx_{:s}_bw{:.2f}_field{:03d}-{:03d}.pdf'
                .format(outdir, st, bw, pt[0], pt[-1]))
