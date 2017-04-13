import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


root_dir = '/Users/piyanat/Google/research/hera1p'
# telescopes = ['hera19', 'hera37', 'hera61', 'hera91',
#               'hera127', 'hera169', 'hera271', 'hera331']
telescope = 'hera331'
# bandwidths = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
bandwidth = 0.08
# groups = ['binning', 'windowing']
group = 'binning'
nfields = 5
field_indexes = np.random.randint(0, 200, 5)

pn = pd.read_hdf('{:s}/stats/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                 .format(root_dir, telescope, telescope, bandwidth, group))
fi, zi, xi = np.genfromtxt('{:s}/interp_delta_21cm_f_z_xi.csv'.format(root_dir),
                           delimiter=',', unpack=True)

plt.close()
fig, ax = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(4.25, 4))
for i in range(len(field_indexes)):
    y0, y1, y2 = pn.iloc[field_indexes[i]][['var', 'skew', 'kurt']].values.T
    ax[0].plot(fi, y0, alpha=0.5)
    ax[1].plot(fi, y1, alpha=0.5)
    ax[2].plot(fi, y2, alpha=0.5)
ax[0].set_xlim(fi[0], fi[-1])
ax[0].set_ylabel('Variance [mK]')
ax[1].set_ylabel('Skewness')
ax[2].set_ylabel('Kurtosis')
ax[2].set_xlabel('Frequency [MHz]')

axt = ax[0].twiny()
axt.set_xlim(ax[0].get_xlim())
axt_ticklabels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
axt_ticklocs = np.interp(axt_ticklabels, xi, fi)
axt.set_xticks(axt_ticklocs)
axt.set_xticklabels(axt_ticklabels)
axt.set_xlabel('Ionized Fraction')
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()