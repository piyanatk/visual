import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator, FixedLocator
import pandas
from astropy.io import fits
from astropy.wcs import WCS


# Read data
root_dir = '/Users/piyanat/Google/research/pdf_paper/'
f, z, xi = np.genfromtxt(root_dir + 'interp_delta_21cm_f_z_xi.csv',
                             delimiter=',', unpack=True)
p4d = pandas.read_hdf(root_dir + 'pdf/pdf_p4d.h5')
model_map = fits.getdata(root_dir + 'maps/model_0.000h_158.195MHz.fits')
fhd_map = fits.getdata(root_dir + 'maps/mwa128_fhd_0.000h_158.195MHz.fits')
gauss_map = fits.getdata(root_dir + 'maps/mwa128_gauss_0.000h_158.195MHz.fits')
res_map = fits.getdata(root_dir + 'maps/mwa128_res_0.000h_158.195MHz.fits')
mask = fits.getdata(root_dir + 'maps/mwa128_mask_0.000h_158.195MHz.fits')\
    .astype(bool)
hdr = fits.getheader(root_dir + 'maps/model_0.000h_158.195MHz.fits')
wcs = WCS(hdr)

# Init fig
gs0 = GridSpec(2, 1, hspace=0.3, wspace=0)
gs1 = GridSpecFromSubplotSpec(2, 5, height_ratios=(0.05, 1),
                              width_ratios=(1, 0.1, 1, 1, 1), wspace=0, hspace=0,
                              subplot_spec=gs0[0])
gs2 = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1])
fig = plt.figure(figsize=(10, 6))
imax = list()
cax = list()
imax.append(fig.add_subplot(gs1[1, 0], projection=wcs))
imax.append(fig.add_subplot(gs1[1, 2], sharex=imax[0], sharey=imax[0],
            projection=wcs))
imax.append(fig.add_subplot(gs1[1, 3], sharex=imax[0], sharey=imax[0],
            projection=wcs))
imax.append(fig.add_subplot(gs1[1, 4], sharex=imax[0], sharey=imax[0],
            projection=wcs))
cax.append(fig.add_subplot(gs1[0, 0]))
cax.append(fig.add_subplot(gs1[0, 2:]))
pax = fig.add_subplot(gs2[0])

# Plot images
model_im = imax[0].imshow(model_map, origin='lower')
fhd_im = imax[1].imshow(fhd_map, vmin=-1, vmax=1, origin='lower')
gauss_im = imax[2].imshow(gauss_map, vmin=-1, vmax=1, origin='lower')
res_im = imax[3].imshow(res_map, vmin=-1, vmax=1, origin='lower')
for ax in imax:
    ax.contour(mask, 1, color='black')
lim = wcs.wcs.crpix + (10 / wcs.wcs.cdelt)
imax[0].set_xlim(*lim)
imax[0].set_ylim(*lim)
imax[0].set_ylabel('Declination [deg]')
imax[1].text(0.6, -0.2, 'Right Ascension [deg]',
             transform=imax[1].transAxes)
# imax[0].coords[0].set_major_formatter('mm')
# ax[0].coords[1].set_major_formatter('dd:mm:ss')
imax[1].coords[1].set_ticklabel_visible(False)
imax[2].coords[1].set_ticklabel_visible(False)
imax[3].coords[1].set_ticklabel_visible(False)

# Colorbar
cbar = list()
cbar.append(fig.colorbar(model_im, cax=cax[0], orientation='horizontal'))
cbar.append(fig.colorbar(fhd_im, cax=cax[1], orientation='horizontal'))
for cb in cbar:
    cb.ax.tick_params(labelbottom='off', labeltop='on')
    cb.ax.xaxis.set_label_position('top')
    cb.locator = MaxNLocator(6)
    cb.update_ticks()
cbar[1].ax.text(0.15, 5.0, 'Brightness Temperature [mK]',
                verticalalignment='top')

# Plot PDF
keys = ['model', 'fhd_scaled', 'gauss', 'res', ]
linestyles = ['y-', 'b-', 'g--', 'r:', ]
labels = ['Model', 'FHD', 'Gaussian', 'Residual']
for k, ls, lb in zip(keys, linestyles, labels):
    x, y = p4d.ix['158.195'][k].T.values
    pax.plot(x, y, ls, label=lb)
pax.set_xscale('symlog', linthreshx=1)
pax.set_xlim(-15, 10)
pax.set_xlabel('Brightness Temperature [mK]', labelpad=-0.05)
pax.set_ylabel('$dP/dT_b$')
# pax.xaxis.set_major_locator(MaxNLocator(8))
pax.legend(loc='upper right')
pax.text(0.05, 0.9, '$x_i=0.5$\n$z=0.979$\n$\\nu=158.195\\,MHz$',
         verticalalignment='top', horizontalalignment='left',
         transform=pax.transAxes)
pax.xaxis.set_major_locator(FixedLocator([-10, -1, 0, 1, 10]))
minor_locs = np.hstack((range(-15, -10), range(-9, -1), np.arange(-0.9, 0, 0.1),
                        np.arange(0.1, 1.0, 0.1), range(2, 10), range(11, 16)))
pax.xaxis.set_minor_locator(FixedLocator(minor_locs))

# Label images
bbox = dict(boxstyle="round", fc='0.8', alpha=0.8, edgecolor='black')
for ax, lb in zip(imax, labels):
    ax.text(0.03, 0.97, lb, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='top',
            color='black', bbox=bbox, size='small')

# Tidy up
gs0.tight_layout(fig, rect=[0, -0.01, 1, 0.98])
gs0.update(hspace=0.22)
fig.canvas.draw()
fig.savefig('mwa_pdf_maps_xi05.pdf', dpi=200)
plt.close()
