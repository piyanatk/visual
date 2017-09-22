import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter


# Lightcone map
arr = np.load('/Users/piyanat/Google/research/projects/hera1p/'
               'hera331_lightcone_ms_p000.npy')

dy = 0.115 / 2  # pixel size on the angular (y) dimension
y = np.arange(-dy * 128, dy * 128, dy)

# x-axis coordinates for plotting and axes transformations.
fi, zi, xi = np.genfromtxt(
    '/Users/piyanat/Google/research/projects/hera1p/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True)

plt.close()
gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 0.02], wspace=0.01)
fig = plt.figure(figsize=(8, 2.5))
ax = fig.add_subplot(gs[0])

im = ax.pcolorfast(fi, y, arr)
ax.set_xlabel('Observed Frequency [MHz]')
ax.set_ylabel('Declination')
ax.yaxis.set_major_formatter(FormatStrFormatter('$%-d^{\circ}$'))

axt = ax.twiny()
axt.set_xlim(ax.get_xlim())
axt_xticklabels = np.arange(3, 10) * 0.1
axt_xticklocs = np.interp(axt_xticklabels, xi, fi)
axt.set_xticks(axt_xticklocs)
axt.set_xticklabels(axt_xticklabels)
axt.set_xlabel('Ionized Fraction')

cax = fig.add_subplot(gs[1])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Brightness\nTemperature [mK]')

fig.subplots_adjust(left=0.08, right=0.91, bottom=0.2, top=0.65)

fig.savefig('/Users/piyanat/Google/research/projects/hera1p/plots_v2/'
            'lightcone_ms_hera331_p000.pdf')
