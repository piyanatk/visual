import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
stats = ['var', 'skew', 'kurt']
prefixes = ['mwa128_fhd', 'mwa128_gauss', 'hera37_gauss', 'hera61_gauss',
            'hera91_gauss', 'hera127_gauss', 'hera169_gauss', 'hera217_gauss',
            'hera271_gauss', 'hera331_gauss']
suffixes = ['windowing', 'binning']
freqs = MWA_FREQ_EOR_ALL_80KHZ
bandwidths = np.arange(1, 16)

x = np.hstack((MWA_FREQ_EOR_ALL_80KHZ, MWA_FREQ_EOR_ALL_80KHZ[-1] + 0.08)) - 0.04
y = np.hstack((bandwidths, 16)) - 0.5

for st in stats:
    for p in prefixes:
        for sf in suffixes:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            df = [pd.read_hdf('{:s}/{:s}_stats_df_bw{:.02f}MHz_{:s}.h5'
                              .format(stats_dir, p, bw, sf)) for bw in bandwidths]
            snr = cal_snr(df, st)
            vmin = 1e-4
            vmax = 1e4
            im = ax.pcolormesh(
                x, y, snr.T,
                norm=MidpointLogNorm(vmin=vmin, vmax=vmax, midpoint=1),
                edgecolors='face', cmap='bwr'
            )
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y[0], y[-1])

            ax.set_xlabel('Observed Frequency [MHz]')
            ax.set_ylabel('Window/Bin Size [MHz]')
            cbar = fig.colorbar(im, orientation='vertical', label='SNR')
            fig.suptitle(('{:s} {:s} {:s} {:s}'
                         .format(p.split('_')[0], p.split('_')[1], st, sf))
                         .upper())
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig('{:s}/snr_color_chart_{:s}_{:s}_{:s}.pdf'
                        .format(stats_dir, p, st, sf), dpi=200)
            plt.close()
