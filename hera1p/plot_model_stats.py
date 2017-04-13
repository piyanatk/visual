import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

root_dir = '/Users/piyanat/Google/research/hera1p'
fi, zi, xi = np.genfromtxt('{:s}/interp_delta_21cm_f_z_xi.csv'.format(root_dir),
                           delimiter=',', unpack=True)
df = pd.read_hdf('{:s}/stats/model/df_stats_hpx_interp_delta_21cm_l128.h5'
                 .format(root_dir))
fig = plt.figure(figsize=(4.5, 4))
ax1 = fig.add_subplot(111)
ax1.plot(fi, df['var'], 'k--')
ax1.plot(fi, df['skew'], 'k-.')
ax1.plot(fi, df['kurt'], 'k:')
ax1.set_xlim(fi[0], fi[-1])
ax1.set_xlabel('Frequency [MHz]')
ax1.set_ylabel('Statistical Values')
ax2 = ax1.twinx()
ax2.plot(fi, xi, 'k-')
ax2.set_xlim(fi[0], fi[-1])
ax2.set_ylabel('Ionized Fraction')

axt = ax1.twiny()
axt.set_xlim(ax1.get_xlim())
axt_ticklabels = [9., 8.5, 8., 7.5, 7., 6.5]
axt_ticklocs = np.interp(axt_ticklabels, zi[::-1], fi[::-1])
axt.set_xticks(axt_ticklocs)
axt.set_xticklabels(axt_ticklabels)
axt.set_xlabel('Redshift')
fig.tight_layout(pad=0.5)
handles = [Line2D([], [], ls='-', c='k'), Line2D([], [], ls='--', c='k'),
           Line2D([], [], ls='-.', c='k'), Line2D([], [], ls=':', c='k')]
labels = ['Ionized Fraction', 'Variance [mK]', 'Skewness', 'Kurtosis']
fig.legend(handles=handles, labels=labels, loc=(0.31, 0.5),
           fontsize='small', handlelength=2.5)
fig.savefig('{:s}/plots/model_stats.pdf'.format(root_dir))
