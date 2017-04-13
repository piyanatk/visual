from glob import glob

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.pyplot import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


def get_data(infile, stat):
    df = pd.read_hdf(infile)
    x = df.index
    y = df[stat].values
    yerr_min = y - df[stat + '_err'].values
    yerr_max = y + df[stat + '_err'].values
    return x, y, yerr_min, yerr_max


if __name__ == '__main__':
    # Figure parameters
    nrows = 15
    ncols = 8
    figsize = (24, 24)
    # Plots and labels parameters
    statistics = ['var', 'skew', 'kurt', 'skew_norm', 'kurt_norm']
    ylabels = dict(var='Variance $(\sigma^2)$ [mK$^2$]',
                   skew='Skewness $(m_3 / \sigma^3)$',
                   kurt='Excess Kurtosis $(m_4 / \sigma^4 - 3)$')
    xlims = (139.915, 195.235)
    ylims = dict(var=(-0.1, 1.0), skew=(-1, 1.5), kurt=(-1, 2.5))
    bandwidths = np.arange(1, 16)  # rows
    telescopes = ['hera37', 'hera61', 'hera91', 'hera127',
                  'hera169', 'hera217', 'hera271', 'hera331']  # cols
    axlabels = np.array(
        ['{:s} {:d} MHz'.format(tel, bw)
         for tel in telescopes for bw in bandwidths]
    )
    axlabels.shape = (nrows, ncols)
    bbox = dict(boxstyle="round", fc="none")

    # Legend parameters
    handlers = [
        Line2D([], [], linestyle='--', color='black', linewidth=1),
        Line2D([], [], linestyle='-', color='black', linewidth=1),
        Patch(color='0.7'),
        Patch(color='0.55')
    ]
    labels = ['Windowing', 'Binning', 'Error (Windowing)', 'Error (Binning)']

    # Get the list of input files
    stats_dir = '/Users/piyanat/research/pdf_paper/new_stats/'
    df_files_window = np.array(
        [stats_dir + '{:s}_gauss_stats_df_bw{:.2f}MHz_windowing.h5'
            .format(tel, bw) for bw in bandwidths for tel in telescopes ]
    )
    df_files_bin = np.array(
        [stats_dir + '{:s}_gauss_stats_df_bw{:.2f}MHz_binning.h5'
            .format(tel, bw) for bw in bandwidths for tel in telescopes]
    )
    df_files_window.shape = (nrows, ncols)
    df_files_bin.shape = (nrows, ncols)
    # print(df_files_bin)

    # Loop over stats and plot
    for stat in statistics:
        # Init figure
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                               figsize=figsize,
                               gridspec_kw=dict(wspace=0, hspace=0))

        # Plot spectral varying cases
        for i in range(nrows):
            for j in range(ncols):
                x1, y1, y1err_min, y1err_max = get_data(
                    df_files_window[i, j], stat
                )
                x2, y2, y2err_min, y2err_max = get_data(
                    df_files_bin[i, j], stat
                )
                ax[i, j].fill_between(x1, y1err_min, y1err_max, color='0.7')
                ax[i, j].fill_between(x2, y2err_min, y2err_max, color='0.55')
                ax[i, j].plot(x1, y1, 'k--', linewidth=1)
                ax[i, j].plot(x2, y2, 'k-', linewidth=1)
                if i == nrows - 1:
                    ax[i, j].set_xlabel(telescopes[j])
                if j == 0:
                    ax[i, j].set_ylabel('{:d} MHz'.format(bandwidths[i]))
                # ax[i, j].text(0.03, 0.8,  axlabels[i, j],
                #               transform=ax[i, j].transAxes, bbox=bbox)

                ax[i, j].grid()
        ax[0, 0].set_xlim(*xlims)
        ax[0, 0].set_ylim(*ylims[stat])
        # Axes labels
        fig.text(0.01, 0.5, ylabels[stat], rotation='vertical',
                 horizontalalignment='left', verticalalignment='center')
        fig.text(0.5, 0.01, 'Frequency [MHz]', horizontalalignment='center',
                 verticalalignment='bottom')
        # fig.text(0.5, 0.99, 'Ionized Fraction', horizontalalignment='center',
        #          verticalalignment='top')

        # Legend
        plt.figlegend(handles=handlers, labels=labels, loc='upper center',
                      ncol=4, fontsize='medium')

        # Tidy up
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        fig.canvas.draw()
        # plt.show()
        fig.savefig(stats_dir + 'heraXX_{:s}.pdf'.format(stat), dpi=200)
        plt.close()
