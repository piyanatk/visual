import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


stats_dir = '/Users/piyanat/Google/research/pdf_paper/maps/'
maps, hdr = fits.getdata(stats_dir + 'hera331_lightcone_slice.fits',
                         header=True)
f, z, xi = np.genfromtxt(stats_dir + 'interp_delta_21cm_f_z_xi.csv',
                         delimiter=',', unpack=True)
gs = GridSpec(2, 1, height_ratios=(0.05, 1), hspace=0, wspace=0)
fig = plt.figure()
cax = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])
im = ax.imshow(maps, extent=(-17.5, 17.5, f[-1], f[0]), aspect=0.5,
               interpolation='spline16')

axt = ax.twinx()
axt.set_ylim(xi[-1], xi[0])
ax.set_xlabel('Angular Distance [deg]')
ax.set_ylabel('Frequency [MHz]')
axt.set_ylabel('Ionized Fraction')

ax.spines['right'].set_visible(False)
axt.spines['bottom'].set_visible(False)
axt.spines['top'].set_visible(False)
axt.spines['left'].set_visible(False)

cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.ax.xaxis.set_label_position('top')
cb.ax.xaxis.set_ticks_position('top')
cb.ax.set_xlabel('Brightness Temperature [mK]')
plt.tight_layout()
fig.savefig('hera331_lightcone.pdf', dpi=200)