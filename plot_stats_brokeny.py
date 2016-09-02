from glob import glob

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator, LogLocator
import pandas


def get_data(panel):
    pn = pandas.read_hdf(panel)
    x = pn.major_axis
    y = pn['fhd_scaled', :, stat].values
    y0 = pn['gauss', :, stat].values
    yerr_mwa_max = y + pn['fhd_scaled', :, 'mwa_'+stat+'_err'].values
    yerr_mwa_min = y - pn['fhd_scaled', :, 'mwa_'+stat+'_err'].values
    yerr_hera_max = y + pn['fhd_scaled', :, 'hera_'+stat+'_err'].values
    yerr_hera_min = y - pn['fhd_scaled', :, 'hera_'+stat+'_err'].values
    return x, y0, y, yerr_mwa_max, yerr_mwa_min, yerr_hera_max, yerr_hera_min


def brokeny_plot_from_gridspec(panel, gs, fig, xoff=False, yoff=False,
                               bottom_ylim=(-0.1, 0.5), top_ylim=(10, 1e4),
                               bottom_yscale='linear', top_yscale='log'):
    x, y0, y, yerr_mwa_max, yerr_mwa_min, yerr_hera_max, yerr_hera_min = \
        get_data(panel)
    # print(panel)
    # print(yerr_mwa_max)
    # print(yerr_mwa_min)
    sgs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs,
                                  height_ratios=[0.5, 1], hspace=0.05)
    ax0 = fig.add_subplot(sgs[0])
    ax1 = fig.add_subplot(sgs[1], sharex=ax0)

    # Plot
    ax1.axhline(0, color='black', linestyle=':')
    ax1.plot(x, y0, 'g--', label='Gaussian')
    ax1.plot(x, y, 'b-', label='FHD')
    ax1.fill_between(x, yerr_mwa_min, yerr_mwa_max, interpolate=True,
                     color='lightslategray', alpha=0.5)
    ax1.fill_between(x, yerr_hera_min, yerr_hera_max, interpolate=True,
                     color='slategray', alpha=0.5)
    ax0.fill_between(x, yerr_mwa_min, yerr_mwa_max, interpolate=True,
                     color='lightslategray', alpha=0.5)
    ax0.fill_between(x, yerr_hera_min, yerr_hera_max, interpolate=True,
                     color='slategray', alpha=0.5)

    # Set different zoom/limit and scaling on the two plots
    ax0.set_yscale(top_yscale)
    ax1.set_yscale(bottom_yscale)
    ax0.set_ylim(yerr_mwa_max.min(), yerr_mwa_max.max())
    # ax0.set_ylim(10, 1e8)
    ax1.set_ylim(*bottom_ylim)
    ax0.set_xlim(138.995, 194.755)

    # Hide the spines between ax0 and ax1
    ax0.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax0.xaxis.tick_top()
    ax0.tick_params(labeltop='off')  # do not put tick labels at the top
    ax1.xaxis.tick_bottom()

    # Add cut-out diagonal lines
    d = .02  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False)
    ax0.plot((-d, +d), (-d*2, +d*2), **kwargs)        # top-left diagonal
    ax0.plot((1 - d, 1 + d), (-d*2, +d*2), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Manage tick labels
    if xoff:
        ax1.xaxis.set_ticklabels([])
    if yoff:
        ax0.yaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
    # ax0.yaxis.set_major_locator(LogLocator(base=10))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    fig.canvas.draw()
    return ax0, ax1


if __name__ == '__main__':
    stat = 'var'
    pn_files = np.vstack(
        (np.sort(glob('/Users/piyanat/research/stats_panels/stats_panel_bw*MHz.h5')),
        np.sort(glob('/Users/piyanat/research/stats_panels/stats_panel_bw*MHz_average.h5')))
    ).T
    gs = GridSpec(4, 2, wspace=0.2, hspace=0.1)
    fig = plt.figure(figsize=(8, 10))
    ax = np.empty((4, 2), dtype=object)
    for i in range(4):
        if i != 3:
            xoff = True
        else:
            xoff = False
        for j in range(2):
            if j == 1:
                yoff = True
            else:
                yoff = False
            ax[i, j] = brokeny_plot_from_gridspec(
                pn_files[i, j], gs[i, j], fig, xoff=xoff, yoff=False
            )
    ax[0, 0][0].set_xlabel('Stack')
    ax[0, 0][0].xaxis.set_label_position('top')
    ax[0, 1][0].set_xlabel('Average')
    ax[0, 1][0].xaxis.set_label_position('top')
    ax[0, 1][1].set_ylabel('1MHz')
    ax[0, 1][1].yaxis.set_label_coords(1.1, 0.8)
    ax[1, 1][1].set_ylabel('2MHz')
    ax[1, 1][1].yaxis.set_label_coords(1.1, 0.8)
    ax[2, 1][1].set_ylabel('6MHz')
    ax[2, 1][1].yaxis.set_label_coords(1.1, 0.8)
    ax[3, 1][1].set_ylabel('8MHz')
    ax[3, 1][1].yaxis.set_label_coords(1.1, 0.8)
    # x, y0, y, yerr_mwa_max, yerr_mwa_min, yerr_hera_max, yerr_hera_min = \
    #     get_data(pn_files[0, 0])
    # ax[0, 0][1].fill_between(x, yerr_mwa_min, yerr_mwa_max, interpolate=True,
    #                          color='lightslategray', alpha=0.5)
    fig.canvas.draw()
    fig.savefig(stat + '.pdf', dpi=200)
