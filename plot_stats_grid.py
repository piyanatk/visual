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


xi = np.genfromtxt(
    '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', usecols=(2,), unpack=True)


def get_data(dataframe):
    df = pd.read_hdf(dataframe)
    x = df.index
    y = df[stat].values
    yerr_min = y - df[stat + '_err'].values
    yerr_max = y + df[stat + '_err'].values
    return x, y, yerr_min, yerr_max


def plot_data(l, m, mwa_file, hera37_file, hera331_file):
    ax[l, m] = fig.add_subplot(gs[l, m])
    print(l, m, mwa_file, hera37_file, hera331_file)
    ax[l, m].axhline(0, linestyle='-', color='lightgray')
    x1, y1, y1err_min, y1err_max = get_data(mwa_file)
    x2, y2, y2err_min, y2err_max = get_data(hera37_file)
    x3, y3, y3err_min, y3err_max = get_data(hera331_file)
    ax[l, m].fill_between(x1, y1err_min, y1err_max, color='0.85')
    ax[l, m].fill_between(x2, y2err_min, y2err_max, color='0.7')
    ax[l, m].fill_between(x3, y3err_min, y3err_max, color='0.55')
    ax[l, m].plot(x1, y1, 'k:', linewidth=1)
    ax[l, m].plot(x2, y2, 'k--', linewidth=1)
    ax[l, m].plot(x3, y3, 'k-', linewidth=1)
    ax[l, m].set_xlim(*xlims)
    ax[l, m].set_ylim(*ylims[stat])
    ax[l, m].xaxis.set_major_locator(MaxNLocator(6, prune='upper'))
    ax[l, m].yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
    ax[l, m].spines['top'].set_visible(False)
    ax[l, m].xaxis.set_ticks_position('bottom')
    ax[l, m].yaxis.set_ticks_position('both')
    ax[l, m].minorticks_on()
    # ax[l, m].spines['right'].set_visible(False)
    # ax[l, m].tick_params(axis='x', which='both', top='off')

    # Add twin axes at the top
    axt[l, m] = ax[l, m].twiny()
    axt[l, m].set_xlim(xi[0], xi[-1])
    axt[l, m].spines['bottom'].set_visible(False)
    axt[l, m].spines['left'].set_visible(False)
    axt[l, m].spines['right'].set_visible(False)
    axt[l, m].xaxis.set_ticks_position('top')
    axt[l, m].minorticks_on()
    # axt[l, m].tick_params(axis='x', which='both', bottom='off')

    # Turn off axis
    if l != nrows-1:
        ax[l, m].xaxis.set_ticklabels([])
    if m == 1:
        ax[l, m].yaxis.set_ticklabels([])
    if l == 0 and m == 0:
        pass
    elif l == 1 and m == 1:
        pass
    else:
        axt[l, m].xaxis.set_ticklabels([])

    # Label subplot
    ax[l, m].text(0.03, 0.8,  axlabels[l, m], transform=ax[l, m].transAxes,
                  bbox=bbox)


if __name__ == '__main__':
    # Figure parameters
    nrows = 5
    ncols = 2
    figsize = (8, 8)

    # Plots and labels parameters
    statistics = ['var', 'skew', 'kurt']
    ylabels = dict(var='Variance $(\sigma^2)$ [mK$^2$]',
                   skew='Skewness $(m_3 / \sigma^3)$',
                   kurt='Excess Kurtosis $(m_4 / \sigma^4 - 3)$')
    xlims = (139.915, 195.235)
    ylims = dict(var=(-0.1, 1.0), skew=(-1, 1.5), kurt=(-1, 2.5))
    bandwidths = [2, 3, 4, 8]
    axlabels = np.array(
        [['80 kHz Window'] + ['{:d} MHz Window'.format(bw) for bw in bandwidths],
         [''] + ['{:d} MHz Bin'.format(bw) for bw in bandwidths]],
    ).T
    print(axlabels.shape)
    bbox = dict(boxstyle="round", fc="none")

    # Legend parameters
    handlers = [
        Line2D([], [], linestyle=':', color='black', linewidth=1),
        Line2D([], [], linestyle='--', color='black', linewidth=1),
        Line2D([], [], linestyle='-', color='black', linewidth=1),
        Patch(color='0.85'),
        Patch(color='0.7'),
        Patch(color='0.55')
    ]
    labels = ['MWA128', 'HERA37', 'HERA331',
              'MWA128 Error', 'HERA37 Error', 'HERA331 Error']

    # Get the list of input files
    stats_dir = '/Users/piyanat/research/pdf_paper/new_stats/'
    df_files_mwa = np.array(
        [[stats_dir + 'mwa128_gauss_stats_df_bw{:.2f}MHz_windowing.h5'.format(bw) for bw in bandwidths],
         [stats_dir + 'mwa128_gauss_stats_df_bw{:.2f}MHz_binning.h5'.format(bw) for bw in bandwidths]]
    ).T
    df_files_hera37 = np.array(
        [[stats_dir + 'hera37_gauss_stats_df_bw{:.2f}MHz_windowing.h5'.format(bw) for bw in bandwidths],
         [stats_dir + 'hera37_gauss_stats_df_bw{:.2f}MHz_binning.h5'.format(bw) for bw in bandwidths]]
    ).T
    df_files_hera331 = np.array(
        [[stats_dir + 'hera331_gauss_stats_df_bw{:.2f}MHz_windowing.h5'.format(bw) for bw in bandwidths],
         [stats_dir + 'hera331_gauss_stats_df_bw{:.2f}MHz_binning.h5'.format(bw) for bw in bandwidths]]
    ).T
    df_files_raw_mwa = stats_dir + 'mwa128_gauss_stats_df_bw0.08MHz_windowing.h5'
    df_files_raw_hera37 = stats_dir + 'hera37_gauss_stats_df_bw0.08MHz_windowing.h5'
    df_files_raw_hera331 = stats_dir + 'hera331_gauss_stats_df_bw0.08MHz_windowing.h5'

    # Loop over stats and plot
    for stat in statistics:
        # Init figure
        gs = GridSpec(nrows=nrows, ncols=ncols, wspace=0, hspace=0)
        fig = plt.figure(figsize=figsize)
        ax = np.empty((nrows, ncols), dtype=object)
        axt = np.empty((nrows, ncols), dtype=object)

        # Plot raw data
        plot_data(0, 0, df_files_raw_mwa, df_files_raw_hera37,
                  df_files_raw_hera331)

        # Plot spectral varying cases
        for i in range(1, nrows):
            for j in range(ncols):
                plot_data(i, j, df_files_mwa[i-1, j], df_files_hera37[i-1, j],
                          df_files_hera331[i-1, j])

        # Axes labels
        fig.text(0.01, 0.5, ylabels[stat], rotation='vertical',
                 horizontalalignment='left', verticalalignment='center')
        fig.text(0.5, 0.01, 'Frequency [MHz]', horizontalalignment='center',
                 verticalalignment='bottom')
        fig.text(0.5, 0.99, 'Ionized Fraction', horizontalalignment='center',
                 verticalalignment='top')

        # Legend
        plt.figlegend(handles=handlers, labels=labels, loc=(0.526, 0.82),
                      ncol=2, fontsize='medium')

        # Tidy up
        # fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.tight_layout(rect=[0.01, 0.01, 0.98, 0.99])
        fig.canvas.draw()
        # plt.show()
        fig.savefig(stats_dir + stat + '.pdf', dpi=200)
        plt.close()
