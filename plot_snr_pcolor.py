import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import pandas as pd

from sim.utils import bin_freqs
from sim.settings import MWA_FREQ_EOR_ALL_80KHZ

stats_dir = '/Users/piyanat/research/pdf_paper/stats/'
stats = ['var', 'skew', 'kurt']
telescopes = ['mwa128', 'hera37', 'hera331']
prefixes = ['mwa128_fhd', 'hera37_gauss', 'hera331_gauss']
# type = '_average'
type = ''
freqs = ['{:.3f}'.format(f) for f in MWA_FREQ_EOR_ALL_80KHZ]
bandwidths = ['{:.2f}'.format(i) for i in range(1, 31)]
linewidths = [1, 1, 1, 1]
linestyles = ['-', '--', '-.', ':']
markerstyles = ['None', 'None', 'None', 'None']
colors = ['0', '0', '0', '0']
bbox = dict(boxstyle="round", fc="none")
# freqs, xi = np.genfromtxt(
#     '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
#     delimiter=',', usecols=(0, 2,), unpack=True
# )

x = np.arange(0.5, len(bandwidths))
y = MWA_FREQ_EOR_ALL_80KHZ - 0.04

for p in prefixes:
    df = dict()
    for bw in bandwidths:
        df[bw] = pd.read_hdf(
            stats_dir + '{:s}_stats_df_bw{:s}MHz{:s}.h5'.format(p, bw, type)
        )

    snr_pn = pd.Panel(items=stats, major_axis=freqs, minor_axis=bandwidths,
                      dtype=float)

    for stat in stats:
        for bw in bandwidths:
            snr = (np.abs(df[bw][stat]) / df[bw][stat+'_err']).values
            chls, bincen = bin_freqs(float(bw))
            assert len(snr) == len(chls)
            for i in range(len(chls)):
                idx = ['{:.3f}'.format(f) for f in chls[i]]
                snr_pn.ix[stat, idx, bw] = snr[i]

        im = plt.pcolormesh(y, x, np.log10(snr_pn[stat].values).T,
                            edgecolors='face', cmap='gray')
        plt.ylim(x[0], x[-1])
        plt.xlim(y[0], y[-1])
        plt.minorticks_on()
        plt.ylabel('Windowing bandwidth [MHz]')
        plt.xlabel('Frequency [MHz]')
        plt.title('{:s} SNR {:s}'.format(stat.upper(), p.upper()))
        cb = plt.colorbar(im, label='Log10(SNR)')
        # cb.set_ticklabels('SNR')
        plt.tight_layout()
        plt.draw()
        plt.savefig('snr_color_chart_{:s}_{:s}_windowing.pdf'.format(p, stat, type), dpi=200)
        plt.close()
    snr_pn.to_hdf(stats_dir + '{:s}_snr_pn.h5'.format(p), 'snr')

# fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False,
#                        figsize=(10, 8))
# axt = np.empty((3, 3), dtype=object)
# for i in range(3):
#     for j in range(3):
#         for bw, ls, ms, lw, cl in zip(bandwidths, linestyles, markerstyles, linewidths, colors):
#             df = pd.read_hdf(stats_dir + '{:s}_stats_df_bw{:s}MHz_average.h5'
#                              .format(prefixes[j], bw))
#             x = df.index
#             y = np.abs(df[stats[i]]) / df[stats[i]+'_err']
#             ax[i, j].plot(x, y, linestyle=ls, marker=ms, color=cl,
#                           markersize=5, markeredgewidth=1, fillstyle='none',
#                           linewidth=lw, markeredgecolor='black')
#             ax[i, j].set_ylim(0, 1e3)
#             ax[i, j].set_yscale('symlog', linthreshy=1)
#
#         # Customize ticks and limits
#         # ax[i, j].yaxis.set_major_locator(MaxNLocator(prune='upper'))
#         # ax[i, j].yaxis.set_major_locator(LogLocator(numticks=4))
#
#         # ax[i, j].yaxis.set_major_formatter(ScalarFormatter())
#         ax[i, j].set_xlim(freqs[0], freqs[-1])
#         ax[i, j].spines['top'].set_visible(False)
#         ax[i, j].xaxis.set_ticks_position('bottom')
#         ax[i, j].yaxis.set_ticks_position('both')
#         ax[i, j].minorticks_on()
#
#         # Add twin axes at the top
#         axt[i, j] = ax[i, j].twiny()
#         axt[i, j].set_xlim(xi[0], xi[-1])
#         axt[i, j].spines['bottom'].set_visible(False)
#         axt[i, j].spines['left'].set_visible(False)
#         axt[i, j].spines['right'].set_visible(False)
#         axt[i, j].xaxis.set_ticks_position('top')
#         axt[i, j].minorticks_on()
#         if i != 0:
#             axt[i, j].xaxis.set_ticklabels([])
#         ax[i, j].text(0.03, 0.89, telescopes[j], transform=ax[i, j].transAxes,
#                       bbox=bbox)
# # Legend
# handlers = [Line2D([], [], linestyle=st, linewidth=lw, color=cl,
#                    marker=ms, markeredgecolor='black',
#                    markersize=5, markeredgewidth=1, fillstyle='none')
#             for st, ms, lw, cl in zip(linestyles, markerstyles, linewidths, colors)]
# labels = ['{:s} MHz'.format(bw) for bw in bandwidths]
# ax[0, 0].legend(handlers, labels, loc='best', fontsize='medium')
#
# # Labels
# ax[0, 0].set_ylabel('Variance SNR')
# ax[1, 0].set_ylabel('Skewness SNR')
# ax[2, 0].set_ylabel('Kurtosis SNR')
# ax[2, 1].set_xlabel('Frequency [MHz]')
# axt[0, 1].set_xlabel('Ionized Fraction')
# ax[0, 0].set_xlim(freqs[0], freqs[-1])
# # fig.text(0.18, 1, 'MWA128', horizontalalignment='left', verticalalignment='top')
# # fig.text(0.5, 1, 'HERA37', horizontalalignment='left', verticalalignment='top')
# # fig.text(0.82, 1, 'HERA331', horizontalalignment='left', verticalalignment='top')
#
# # Tidy up
# # plt.show()
# plt.tight_layout()
# fig.savefig('snr.pdf', dpi=200)
# plt.close()
