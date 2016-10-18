import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd

from sim.utils import bin_freqs
from sim.settings import MWA_FREQ_EOR_ALL_80KHZ


class MidpointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        LogNorm.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = np.log10([self.vmin, self.midpoint, self.vmax])
        y = [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log10(value), x, y))


def cal_snr(df, stat):
    snr = np.empty((len(freqs), len(bandwidths)))
    for bw in bandwidths:
        chls, bincen = bin_freqs(bw, freqs_list=range(len(freqs)))
        tmp = (np.abs(df[bw-1][stat]) / df[bw-1][stat+'_err']).values
        for binnum in range(len(chls)):
            snr[chls[binnum], bw-1] = tmp[binnum]
    return snr


stats_dir = '/Users/piyanat/research/pdf_paper/new_stats/'
stats = ['skew', 'kurt']
telescopes = ['mwa128', 'hera37', 'hera61', 'hera91', 'hera127', 'hera169',
              'hera217', 'hera271', 'hera331']
suffixes = ['windowing', 'binning']
freqs = MWA_FREQ_EOR_ALL_80KHZ
bandwidths = np.arange(1, 16)
linewidths = [1, 1, 1, 1]
linestyles = ['-', '--', '-.', ':']
markerstyles = ['None', 'None', 'None', 'None']
colors = ['0', '0', '0', '0']
bbox = dict(boxstyle="round", fc="none")
vlim = dict(mwa128=(3.95831814817e-10, 1.34505273536),
            hera37=(9.08922761215e-08, 58.5674950271),
            hera331=(0.00201022967135, 687.002187048))

x = np.hstack((MWA_FREQ_EOR_ALL_80KHZ, MWA_FREQ_EOR_ALL_80KHZ[-1] + 0.08)) - 0.04
y = np.hstack((bandwidths, 16)) - 0.5

nrows = 2
ncols = 2

for t in telescopes:
    gs0 = GridSpec(1, 2, width_ratios=[1, 0.02])
    gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0], wspace=0.05, hspace=0.05)
    fig = plt.figure()
    ax = np.empty((nrows, ncols), dtype='object')

    for i in range(nrows):
        for j in range(ncols):
            ax[i, j] = fig.add_subplot(gs[i, j])
    cax = fig.add_subplot(gs0[1])

    for j in range(ncols):
        df = [pd.read_hdf(stats_dir + '{:s}_gauss_stats_df_bw{:.02f}MHz_{:s}.h5'
                          .format(t, bw, suffixes[j])) for bw in bandwidths]
        for i in range(nrows):
            snr = cal_snr(df, stats[i])
            # print(snr.shape, x.size, y.size)
            # vmin = vlim[t][0]
            # vmax = vlim[t][1]
            vmin = 1e-4
            vmax = 1e4
            im = ax[i, j].pcolormesh(
                x, y, snr.T,
                norm=MidpointLogNorm(vmin=vmin, vmax=vmax, midpoint=1),
                edgecolors='face', cmap='bwr'
            )
            ax[i, j].set_xlim(x[0], x[-1])
            ax[i, j].set_ylim(y[0], y[-1])
            if i != nrows-1:
                ax[i, j].xaxis.set_ticklabels([])
            if j == 1:
                ax[i, j].yaxis.set_ticklabels([])

    ax[0, 0].xaxis.set_label_position("top")
    ax[0, 1].xaxis.set_label_position("top")
    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.set_label_position("right")
    ax[0, 0].set_xlabel('Windowing')
    ax[0, 1].set_xlabel('Binning')
    ax[1, 0].set_xlabel('Observed Frequency [MHz]', labelpad=10)
    ax[1, 0].xaxis.set_label_coords(1, -0.1)
    ax[0, 1].set_ylabel('Skewness ($S_3$)')
    ax[1, 1].set_ylabel('Kurtosis ($S_4$)')
    ax[1, 0].set_ylabel('Window/Bin Size [MHz]', labelpad=10)
    ax[1, 0].yaxis.set_label_coords(-0.1, 1)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', label='SNR')
    fig.suptitle(t.upper(), x=0.45)

    gs0.tight_layout(fig, rect=[0, 0, 1, 1])
    fig.canvas.draw()
    fig.savefig(stats_dir + 'snr_color_chart_{:s}.pdf'.format(t), dpi=200)
