import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.pyplot import Line2D
# from matplotlib.ticker import MaxNLocator
from astropy.io import fits
from astropy.wcs import WCS


if __name__ == '__main__':
    # Figure parameters
    figsize = (10, 5)

    telescopes = ['hera37', 'hera61', 'hera91', 'hera127',
                  'hera169', 'hera217', 'hera271', 'hera331']
    mapsdir = '/data3/piyanat/runs/post_uvlt50_I_intnorm/maps/'
    mapsfiles = [
        '{:s}{:s}/bin3.00MHz/{:s}_gauss_0.000h_bw3.00MHz_178.555MHz.fits'
        .format(mapsdir, t, t) for t in telescopes
    ]
    maskfiles = [
        '{:s}{:s}/bin3.00MHz/{:s}_mask_0.000h_bw3.00MHz_178.555MHz.fits'
        .format(mapsdir, t, t) for t in telescopes
    ]
    nmaps = len(mapsfiles)
    hdr = fits.getheader(mapsfiles[0])
    wcs = WCS(hdr)
    gs = GridSpec(nrows=1, ncols=nmaps)
    fig = plt.figure(figsize=figsize)
    ax = np.empty(nmaps, dtype=object)
    ax[0] = fig.add_subplot(gs[0], projection=wcs)
    for i in range(1, nmaps):
        ax[i] = fig.add_subplot(gs[i], sharex=ax[0], sharey=ax[0],
                                projection=wcs)
    for i in range(nmaps):
        ax[i].imshow(fits.getdata(mapsfiles[i]), vmin=-1, vmax=1,
                     origin='lower')
        ax[i].contour(fits.getdata(maskfiles[i]), levels=[1, ], colors='black')
    fig.savefig('heraXX_maps_bin3MHz_178.555MHz.pdf')
