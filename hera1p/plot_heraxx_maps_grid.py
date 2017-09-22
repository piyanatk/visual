import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from astropy import units as u


data_dir = '/Users/piyanat/Google/research/projects/hera1p/maps_plot'
telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
xi = [0.5, 0.7, 0.95]

nrows = len(xi)
ncols = len(telescope)
xlabels = ['HERA19', 'HERA37', 'HERA128', 'HERA240 Core', 'HERA350 Core']
ylabels = ['$x_i={:.2f}$'.format(i) for i in xi]

gs = GridSpec(nrows=nrows, ncols=ncols, wspace=0, hspace=0)
plt.close()
fig = plt.figure(figsize=(8, 4.5))
ax = np.empty((nrows, ncols), dtype=object)
for i in range(nrows):
    for j in range(ncols):
        hdul = fits.open('{:s}/{:s}_mc_map_p000_xi{:.2f}.fits'
                         .format(data_dir, telescope[j], xi[i]))
        data = hdul[0].data
        hdr = hdul[0].header
        hdr['CRVAL1'] = 0
        hdr['CRVAL2'] = 0
        w = WCS(hdr)

        mask = np.load('{:s}/{:s}_mask_xi{:.2f}.npy'
                       .format(data_dir, telescope[j], xi[i]))
        data *= 1e3
        data -= data[mask].mean()
        # min = data[mask].min()
        # max = data[mask].max()
        # print(min, max)
        ax[i, j] = fig.add_subplot(gs[i, j], projection=w)
        im = ax[i, j].imshow(data, origin='lower', vmin=-2.5,
                             vmax=2.5)

        lon = ax[i, j].coords[0]
        lat = ax[i, j].coords[1]

        # First turn everything off
        lon.set_ticks_visible(False)
        lon.set_ticklabel_visible(False)
        lat.set_ticks_visible(False)
        lat.set_ticklabel_visible(False)
        lon.set_axislabel('')
        lat.set_axislabel('')

        if i == 0:
            lon.set_axislabel_position('t')
            lon.set_axislabel(xlabels[j])
        if j == ncols - 1:
            lat.set_axislabel_position('r')
            lat.set_axislabel(ylabels[i])
        if i == nrows - 1:
            lon.set_ticks_visible(True)
            lon.set_ticklabel_visible(True)
            lon.set_ticks_position('b')
            # lon.set_major_formatter('-d')
        if j == 0:
            lat.set_ticks_visible(True)
            lat.set_ticklabel_visible(True)
            lat.set_ticks_position('l')


ax[1, 0].coords[1].set_axislabel('Declination')
ax[2, 2].coords[0].set_axislabel('Right Ascension')
gs.tight_layout(fig, w_pad=0, h_pad=0, rect=[0.05, 0.06, 0.88, 0.97])
# fig.subplots_adjust(wspace=0, hspace=0)
cax = fig.add_axes([0.9, 0.095, 0.015, 0.84])
cbar = fig.colorbar(ax[0, 0].images[0], cax=cax)
cbar.set_label('Brightness Temperature [mK]')
# tick_locator = ticker.MaxNLocator(nbins=4)
# cbar.locator = tick_locator
# cbar.update_ticks()
# fig.savefig('{:s}/maps_plot_saturated.pdf'.format(data_dir))
fig.savefig('/Users/piyanat/Google/research/projects/hera1p/plots_v2/'
            'heraxx_maps_grid.pdf'.format(data_dir))
