import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
from matplotlib.pyplot import Line2D
from matplotlib.ticker import MaxNLocator
from astropy.io import fits
from astropy.wcs import WCS
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
    figsize = (24, 10)
    frequencies = ['145.115', '178.555']
    telescopes = ['hera37', 'hera61', 'hera91', 'hera127',
                  'hera169', 'hera217', 'hera271', 'hera331']
    nrows = len(frequencies) + 1
    ncols = len(telescopes)
    # Read files
    mapsdir = '/Users/piyanat/Google/research/pdf_paper/maps'
    mapfiles = np.array(
        ['{:s}/{:s}_gauss_0.000h_bw3.00MHz_{:s}MHz.fits'
         .format(mapsdir, t, f) for f in frequencies for t in telescopes]
    )
    maskfiles = np.array(
        ['{:s}/{:s}_mask_0.000h_bw3.00MHz_{:s}MHz.fits'
         .format(mapsdir, t, f) for f in frequencies for t in telescopes]
    )
    mapfiles.shape = (nrows-1, ncols)
    maskfiles.shape = (nrows-1, ncols)
    # Stat files
    statsdir = '/Users/piyanat/Google/research/pdf_paper/new_stats'
    df_files_window = np.array(
        ['{:s}/{:s}_gauss_stats_df_bw3.00MHz_windowing.h5'
         .format(statsdir, tel) for tel in telescopes]
    )
    df_files_bin = np.array(
        ['{:s}/{:s}_gauss_stats_df_bw3.00MHz_binning.h5'
         .format(statsdir, tel) for tel in telescopes]
    )

    hdr = fits.getheader(mapfiles[0, 0])
    wcs = WCS(hdr)
    # PSF size
    reso = np.vstack(
        (1.22 * (3e8 / 145.115e6) / (14 * ((np.arange(4, 12) * 2) - 1)),
         1.22 * (3e8 / 178.555e6) / (14 * ((np.arange(4, 12) * 2) - 1)))
    )
    reso *= 180 / np.pi
    # More figure params
    lim = wcs.wcs.crpix + (7 / wcs.wcs.cdelt)
    gs0 = GridSpec(nrows=2, ncols=1, height_ratios=(1, 2))
    gs1 = GridSpecFromSubplotSpec(nrows=1, ncols=ncols,
                                  wspace=0.05, hspace=0, subplot_spec=gs0[0])
    gs2 = GridSpecFromSubplotSpec(nrows=nrows-1, ncols=ncols,
                                  wspace=0.05, hspace=0, subplot_spec=gs0[1])
    fig = plt.figure(figsize=figsize)
    ax = np.empty((nrows, ncols), dtype=object)

    # Plot stats
    for j in range(ncols):
        ax[0, j] = fig.add_subplot(gs1[j])
        x1, y1, y1err_min, y1err_max = get_data(df_files_window[j], 'kurt')
        x2, y2, y2err_min, y2err_max = get_data(df_files_bin[j], 'kurt')
        ax[0, j].fill_between(x1, y1err_min, y1err_max, color='0.7')
        ax[0, j].fill_between(x2, y2err_min, y2err_max, color='0.55')
        ax[0, j].plot(x1, y1, 'k--', linewidth=1)
        ax[0, j].plot(x2, y2, 'k-', linewidth=1)
        ax[0, j].set_xlim(x1[0], x1[-1])
        ax[0, j].set_ylim(-1, 2)
        ax[0, j].xaxis.set_major_locator(MaxNLocator(6, prune='both'))
        ax[0, j].grid()
        if j != 0:
            ax[0, j].yaxis.set_ticklabels([])

    # Plot images
    for i in range(1, nrows):
        for j in range(ncols):
            ax[i, j] = fig.add_subplot(gs2[i-1, j], projection=wcs)
            im = ax[i, j].imshow(fits.getdata(mapfiles[i-1, j]),
                                 vmin=-1, vmax=1, origin='lower')
            ax[i, j].contour(fits.getdata(maskfiles[i-1, j]),
                             levels=[0], colors='black')
            ax[i, j].set_xlim(*lim)
            ax[i, j].set_ylim(*lim)
            if i == 1:
                ax[i, j].coords[0].set_ticklabel_visible(False)
            if j != 0:
                ax[i, j].coords[1].set_ticklabel_visible(False)
            if i == nrows-1:
                ax[i, j].set_xlabel(telescopes[j].upper())
            if j == 0:
                ax[i, j].set_ylabel(frequencies[i-1] + ' MHz')
            c = Circle((6, -32), reso[i-1, j], edgecolor='black',
                       facecolor='white', alpha=0.5,
                       transform=ax[i, j].get_transform('fk5'))
            ax[i, j].add_patch(c)

    gs0.tight_layout(fig, rect=[0.015, 0.05, 0.95, 0.97])
    gs0.update(hspace=0.15)

    # fig.suptitle('3 MHz Bin Maps')
    fig.text(0.46, 0.01, 'Right Ascension [deg]')
    fig.text(0.46, 0.63, 'Frequency [MHz]')
    fig.text(0.005, 0.4, 'Declination [deg]', rotation='vertical')
    fig.text(0.005, 0.8, 'Kurtosis', rotation='vertical')

    cbar_ax = fig.add_axes([0.95, 0.1, 0.01, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Brightness Temperature [mK]')

    handlers = [
        Line2D([], [], linestyle='--', color='black', linewidth=1),
        Line2D([], [], linestyle='-', color='black', linewidth=1),
        Patch(color='0.7'),
        Patch(color='0.55')
    ]
    labels = ['Windowing', 'Binning', 'Error (Windowing)', 'Error (Binning)']
    fig.legend(handles=handlers, labels=labels,
               loc='upper center', ncol=4)
    # plt.show()
    fig.savefig('heraXX_stats_maps_bin3MHz_145.115MHz_178.555MHz.pdf')
