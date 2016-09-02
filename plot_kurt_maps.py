from string import ascii_uppercase
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


freqs, xi = np.genfromtxt('/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
                   delimiter=',', usecols=(0, 2), unpack=True)


def read_fits(filenames):
    data = []
    header = []
    for f in filenames:
        hdul = fits.open(f)
        data.append(hdul[0].data)
        header.append(hdul[0].header)
        hdul.close()
    return data, header


def label_plots(xlocs, ylocs):
    label_texts = ascii_uppercase
    bbox = dict(boxstyle="round", fc="0.8")
    for i in range(n_maps):
        stat_ax.axvline(xlocs[i], linestyle=':', color='black', linewidth=0.1)
        stat_ax.text(xlocs[i], ylocs[i], label_texts[i], color='black',
                     horizontalalignment='center', verticalalignment='bottom',
                     bbox=bbox)
        maps_ax.ravel()[i].text(0.05, 0.95, label_texts[i],
                                transform=maps_ax.ravel()[i].transAxes,
                                horizontalalignment='left', verticalalignment='top',
                                color='black', bbox=bbox)


def plot_maps(ax):
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            im = ax[i, j].imshow(maps[k], vmin=-1, vmax=1,
                                 origin='lower', cmap=plt.cm.rainbow)
            ax[i, j].contour(masks[k], 1, colors='black')
            # Labels and limits
            ax[i, j].set_xlim(*maps_lim[k])
            ax[i, j].set_ylim(*maps_lim[k])
            if i == nrows-1:
                ax[i, j].coords[0].set_ticklabel_visible(True)
            else:
                ax[i, j].coords[0].set_ticklabel_visible(False)
            if j == 0:
                ax[i, j].coords[1].set_ticklabel_visible(True)
            else:
                ax[i, j].coords[1].set_ticklabel_visible(False)
    # Colorbar
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.set_xlabel('Brightness Temperature [mK]')
    # fig.savefig(outdir + 'kurt_maps_{:.3f}MHz_{:.3f}MHz_{:.3f}MHz.pdf'
    #             .format(*maps_xlocs), dpi=200)


def plot_stat(ax):
    x = pn.major_axis
    for tel, st, cl in zip(telescopes, styles, colors):
        y = pn[tel][stat]
        yerr_max = y + pn[tel][stat + '_err']
        yerr_min = y - pn[tel][stat + '_err']
        ax.fill_between(x, yerr_min, yerr_max, color=cl)
        ax.plot(x, y, st, linewidth=2)
        if tel == 'hera331':
            ax.axhline(y.std(), color='black', linestyle='-', linewidth=0.1)
            ax.axhline(-y.std(), color='black', linestyle='-', linewidth=0.1)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ylims[stat])
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))

    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()

    # Add twin axes at the top
    axt = ax.twiny()
    axt.set_xlim(xi[0], xi[-1])
    axt.spines['bottom'].set_visible(False)
    axt.spines['left'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axt.xaxis.set_ticks_position('top')
    axt.minorticks_on()

    # Legend
    handlers = [
        Line2D([], [], linestyle=':', color='black', linewidth=2),
        Line2D([], [], linestyle='--', color='black', linewidth=2),
        Line2D([], [], linestyle='-', color='black', linewidth=2),
        Patch(color=colors[0]),
        Patch(color=colors[1]),
        Patch(color=colors[2])
    ]
    labels = [
        'MWA128', 'HERA37', 'HERA331',
        'MWA128 Error', 'HERA37 Error', 'HERA331 Error'
    ]
    # ax.legend(handles=handlers, labels=labels, loc='upper left', ncol=2)

    # Tidy up
    ax.set_xlabel('Frequency [MHz]')
    axt.set_xlabel('Ionized Fraction')


if __name__ == '__main__':
    # Figure parameters
    nrows = 2
    ncols = 3
    # Stat plot parameters
    stat = 'kurt'
    styles = ['k:', 'k--', 'k-']
    ylims = dict(var=(-0.1, 1.0), skew=(-1, 1.5), kurt=(-1, 2.5))
    ylabels = dict(var='Variance $(\sigma^2)$ [mK$^2$]',
                   skew='Skewness $(m_3 / \sigma^3)$',
                   kurt='Excess Kurtosis $(m_4 / \sigma^4 - 3)$')
    colors = ['0.85', '0.70', '0.55']
    # Stat data parameters
    telescopes = ['mwa128', 'hera37', 'hera331']
    stats_dir = '/Users/piyanat/research/pdf_paper/stats/'
    pn = pd.Panel(
        dict(mwa128=pd.read_hdf(stats_dir + 'mwa128_fhd_stats_df_bw3.00MHz_average.h5'),
             hera37=pd.read_hdf(stats_dir + 'hera37_gauss_stats_df_bw3.00MHz_average.h5'),
             hera331=pd.read_hdf(stats_dir + 'hera331_gauss_stats_df_bw3.00MHz_average.h5'))
    )
    # Map data parameters
    maps_dir = '/Users/piyanat/Google/research/pdf_paper/maps/'
    map_files = glob(maps_dir + 'hera331_gauss*.fits')
    mask_files = glob(maps_dir + 'hera331_mask*.fits')
    # Read data
    maps, hdr = read_fits(map_files)
    masks, hdr = read_fits(mask_files)
    n_maps = len(maps)
    wcs = [WCS(hdr[i]) for i in range(n_maps)]
    maps_lim = [wcs[i].wcs.crpix + (6 / wcs[i].wcs.cdelt)
                for i in range(n_maps)]
    # Make axes
    gs0 = GridSpec(nrows=2, ncols=1, height_ratios=[0.8, nrows])
    gs1 = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], wspace=0, hspace=0)
    gs2 = GridSpecFromSubplotSpec(nrows+1, ncols, subplot_spec=gs0[1],
                                  height_ratios=np.hstack(([0.1], np.ones(nrows))),
                                  wspace=0, hspace=0)
    fig = plt.figure(figsize=(8, 8.5))
    stat_ax = fig.add_subplot(gs1[0])
    cax = fig.add_subplot(gs2[0, :])
    maps_ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            if i == 0 and j == 0:
                maps_ax[i, j] = fig.add_subplot(gs2[1+i, j], projection=wcs[k])
            else:
                maps_ax[i, j] = fig.add_subplot(
                    gs2[1+i, j], sharex=maps_ax[0, 0], sharey=maps_ax[0, 0],
                    projection=wcs[k]
                )
    plot_stat(stat_ax)
    plot_maps(maps_ax)
    xlocs = [145.115, 148.155, 163.355, 178.555, 187.675, 193.755]
    ylocs = np.ones_like(xlocs) * 2.0
    label_plots(xlocs, ylocs)
    # fig.text(0.01, (1-1./(nrows+1))*1.2, ylabels[stat],
    #          rotation='vertical', verticalalignment='center')
    # fig.text(0.5, 0.01, 'Right Ascension [$^{\circ}$]',
    #          horizontalalignment='center')
    fig.text(0.01, 1./(nrows+1), 'Declination [$^{\circ}$]',
             rotation='vertical', verticalalignment='center')
    stat_ax.set_ylabel(ylabels[stat])
    maps_ax[1, 1].set_xlabel('Right Ascension [$^{\circ}$]')
    # maps_ax[0, 0].set_ylabel('Declination [$^{\circ}$]')
    gs0.tight_layout(fig, rect=[-0.02, 0.05, 1, 1])
    # plt.tight_layout()
    # plt.show()
    plt.savefig('kurt_maps_hera331_bw3MHz.pdf', dpi=200)