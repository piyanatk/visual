import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits
from astropy.wcs import WCS

root_dir = '/Users/piyanat/Google/data/hera1p'
stats_dir = '{:s}/stats'.format(root_dir)
telescope = 'hera331'
bandwidth = 0.08
group = 'binning'
field = 25

# Load data
ds = xr.open_dataset('{:s}/hera1p_all_stats_bw{:.2f}MHz.nc'
                       .format(stats_dir, bandwidth))
s = ds['single_field_stats'].isel(telescope=8, averaging=0, field=25, stat=2)
s_err = ds['single_field_sample_errors'].isel(telescope=8, averaging=0, stat=2)
s_mean = ds['drift_scan_stats'].isel(telescope=8, averaging=0, stat=2)
x = ds.frequency.values

# Figure out corresponding maps
# print(freq[np.where(kurt > 2)])
# print(kurt[np.where(kurt > 2)])

# Load maps
m1 = fits.getdata('{:s}/{:s}_mc_maps_p026_158.435MHz.fits'.format(root_dir, telescope))
m2 = fits.getdata('{:s}/{:s}_mc_maps_p026_170.035MHz.fits'.format(root_dir, telescope))
m1 -= m1.mean()
m2 -= m2.mean()
m1 *= 1e3
m2 *= 1e3
# Dividing by the sum with equal the variance
# m1 /= m1.sum()
# m2 /= m2.sum()

vmin = np.dstack((m1, m2)).min()
vmax = np.dstack((m1, m2)).max()

# Get WCS for WCSAxes
hdr = fits.getheader('{:s}/{:s}_mc_maps_p026_158.435MHz.fits'.format(root_dir, telescope))
hdr['CRVAL1'] = 0
hdr['CRVAL2'] = 0
w = WCS(hdr)

# Checking if kurtosis values match
# fov1 = 1.22 * (const.si.c.value / 158.435e6) / 14 * 180 / np.pi
# fov2 = 1.22 * (const.si.c.value / 170.035e6) / 14 * 180 / np.pi
# xx = np.arange(-128, 128) * hdr['CDELT2']
# rr = np.sqrt(xx[..., np.newaxis] ** 2 + xx ** 2)
# print(fov1)
# print(kurtosis(m1[rr < fov1/2]))
# print(kurtosis(m2[rr < fov2/2]))

# Plot
plt.close()
gs = GridSpec(2, 4, width_ratios=[1, 1, 0.05, 0.1], height_ratios=[1.5, 1])
fig = plt.figure(figsize=(5.5, 4.75))
max1 = fig.add_subplot(gs[0, 0], projection=w)
max2 = fig.add_subplot(gs[0, 1], projection=w)
cax = fig.add_subplot(gs[0, 2])
dax = fig.add_subplot(gs[1, :])

im = max1.imshow(m1, origin='lower', vmin=vmin, vmax=vmax)
max2.imshow(m2, origin='lower', vmin=vmin, vmax=vmax)
max1.coords[0].set_ticklabel_position('t')
max2.coords[0].set_ticklabel_position('t')
max2.coords[1].set_ticklabel_visible(False)

l1 = dax.plot(x, s, label='Field {:d}'.format(field), zorder=3)
l2 = dax.plot(x, s_mean, 'k:', label='Full sky', zorder=3)
l3 = dax.fill_between(x, s_mean - s_err, s_mean + s_err,
                      label='Single-field sample variance',
                      color='0.5', alpha=0.5, zorder=2)
dax.axhline(0, ls='--', color='k', zorder=0)
dax.set_xlim(x[0], x[-1])
dax.spines['top'].set_visible(False)
dax.set_xlabel('Observed Frequency [MHz]')
dax.set_ylabel('Kurtosis')

gs.tight_layout(fig, rect=[0, 0, 0.95, 0.92])
gs.update(hspace=0.1, wspace=0)
cbar = fig.colorbar(im, cax=cax)

# Draw lines from peaks to spines of images
fx1, fy1 = fig.transFigure.inverted().transform(
    dax.transData.transform([158.435, 3.61])
)
fx2, fy2 = fig.transFigure.inverted().transform(
    dax.transData.transform([170.035, 0.025])
)
m1xl, m1yl = fig.transFigure.inverted().transform(
    max1.transAxes.transform([0, 0])
)
m1xr, m1yr = fig.transFigure.inverted().transform(
    max1.transAxes.transform([1, 0])
)
m2xl, m2yl = fig.transFigure.inverted().transform(
    max2.transAxes.transform([0, 0])
)
m2xr, m2yr = fig.transFigure.inverted().transform(
    max2.transAxes.transform([1, 0])
)
g1 = Line2D([fx1, m1xl], [fy1, m1yl], c='k', ls=':', lw=0.8,
            transform=fig.transFigure, figure=fig, zorder=1)
g2 = Line2D([fx1, m1xr], [fy1, m1yr], c='k', ls=':', lw=0.8,
            transform=fig.transFigure, figure=fig, zorder=1)
g3 = Line2D([fx2, m2xl], [fy2, m2yl], c='k', ls=':', lw=0.8,
            transform=fig.transFigure, figure=fig, zorder=1)
g4 = Line2D([fx2, m2xr], [fy2, m2yr], c='k', ls=':', lw=0.8,
            transform=fig.transFigure, figure=fig, zorder=1)
fig.lines.extend([g1, g2, g3, g4])

lines = [l1[0], l2[0], l3]
labels = [ll.get_label() for ll in lines]
fig.legend(lines, labels, loc='upper center', ncol=3)
fig.savefig('{:s}/plots_v2/kurt_maps_{:s}_field{:03d}_bw{:.2f}MHz_{:s}.pdf'
            .format(root_dir, telescope, field, bandwidth, group))
# plt.show()