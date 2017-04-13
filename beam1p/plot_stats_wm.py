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


def get_data(infile, s):
    df = pd.read_hdf(infile)
    x = df.index
    y = df[s].values
    return x, y


if __name__ == '__main__':
    statistics = ['var', 'skew', 'kurt']
    telescopes = ['hera37', 'hera61', 'hera91', 'hera127',
                  'hera169', 'hera217', 'hera271', 'hera331']
    bandwidths = np.hstack(([0.08], np.arange(1, 9)))

    nrows = len(bandwidths)
    ncols = len(telescopes)

    figsize = (28, 20)
    ylabels = dict(var='Variance $(\sigma^2)$ [mK$^2$]',
                   skew='Skewness $(m_3 / \sigma^3)$',
                   kurt='Kurtosis $(m_4 / \sigma^4 - 3)$')
    xlims = (139.915, 195.235)
    ylims = dict(var=(0, 1), skew=(-1, 1), kurt=(-1, 2.5))
    nticks = dict(var=8, skew=8, kurt=7)
    axlabels = np.array(
        ['{:s}-{:.2f}MHz Windowing'.format(tel, bw)
         for bw in bandwidths for tel in telescopes]
    ).reshape((nrows, ncols))
    bbox = dict(boxstyle="round", fc="none")

    # Legend parameters
    handlers = [
        Line2D([], [], linestyle='--', color='black', linewidth=1),
        Line2D([], [], linestyle='-', color='black', linewidth=1),
    ]
    labels = ['Un-masked', 'Wedge Masked']

    # Get the list of input files
    stats_dir = '/Users/piyanat/research/project/' \
                'instrument_systematic_on_1pt_stats/new_stats'

    # Loop over stats and plot
    for stat in statistics:
        # Init figure
        fig, ax = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, figsize=figsize,
            gridspec_kw=dict(hspace=0, wspace=0)
        )

        nwm_files = np.array(
            ['{:s}/{:s}_gauss_stats_df_bw{:.2f}MHz_windowing.h5'
             .format(stats_dir, tel, i) for i in bandwidths for tel in telescopes]
        ).reshape((nrows, ncols))
        wm_files = np.array(
            ['{:s}/{:s}_gauss_wm_imap_stats_df_bw{:.2f}MHz_windowing.h5'
             .format(stats_dir, tel, i) for i in bandwidths for tel in telescopes]
        ).reshape((nrows, ncols))

        # Plot spectral varying cases
        for i in range(nrows):
            for j in range(ncols):
                x1, y1 = get_data(
                    nwm_files[i, j], stat
                )
                x2, y2 = get_data(
                    wm_files[i, j], stat
                )
                ax[i, j].plot(x1, y1, 'r--', linewidth=1)
                ax[i, j].plot(x2, y2, 'b-', linewidth=1)
                # ax[i, j].set_xlabel('tel')
                # if i == nrows - 1:
                #     ax[i, j].set_xlabel(telescopes[j])
                # if j == 0:
                #     ax[i, j].set_ylabel('{:d} MHz'.format(bandwidths[i]))
                ax[i, j].text(0.05, 0.9,  axlabels[i, j],
                              transform=ax[i, j].transAxes, bbox=bbox)
                ax[i, j].xaxis.set_major_locator(MaxNLocator(6, prune='upper'))
                ax[i, j].yaxis.set_major_locator(MaxNLocator(nticks[stat], prune='upper'))
                ax[i, j].grid()
        ax[0, 0].set_xlim(*xlims)
        ax[0, 0].set_ylim(*ylims[stat])
        # Axes labels
        fig.text(0.01, 0.5, ylabels[stat], rotation='vertical',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize='xx-large')
        fig.text(0.5, 0.01, 'Frequency [MHz]', horizontalalignment='center',
                 verticalalignment='bottom', fontsize='xx-large')
        # fig.text(0.5, 0.99, 'Ionized Fraction', horizontalalignment='center',
        #          verticalalignment='top')

        # Legend
        plt.figlegend(handles=handlers, labels=labels, loc='upper center',
                      ncol=2, fontsize='xx-large')

        # Tidy up
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
        fig.canvas.draw()
        # plt.show()
        fig.savefig('heraXX_wm_imaps_{:s}_windowing.pdf'.format(stat), dpi=200)
        plt.close()
