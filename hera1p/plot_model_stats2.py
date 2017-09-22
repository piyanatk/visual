import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

root_dir = '/Users/piyanat/Google/data/hera1p'
fi, zi, xi = np.genfromtxt('{:s}/interp_delta_21cm_f_z_xi.csv'.format(root_dir),
                           delimiter=',', unpack=True)
df = pd.read_hdf('{:s}/stats/model/df_stats_hpx_interp_delta_21cm_l128.h5'
                 .format(root_dir))
fig = plt.figure(figsize=(5, 3.5))
ax1 = fig.add_subplot(111)
ax1.axhline(0, ls=':', color='k', lw=0.8, zorder=0)
l1 = ax1.plot(fi, df['var'], '-', label='Variance', zorder=1)
l2 = ax1.plot(fi, df['skew'], '--', label='Skewness', zorder=2)
l3 = ax1.plot(fi, df['kurt'], '-.', label='Kurtosis', zorder=3)

# Lower x-axis
ax1.set_xlim(fi[0], fi[-1])
ax1.set_xlabel('Observed Frequency [MHz]')
ax1.set_ylabel('Statistical Values')

# Make upper x-axis with redshift coordinates
axt_xticklabels = np.arange(3, 10) * 0.1
axt_xticklocs = np.interp(axt_xticklabels, xi, fi)
axt = ax1.twiny()
axt.set_xlim(ax1.get_xlim())
axt.set_xticks(axt_xticklocs)
axt.set_xticklabels(axt_xticklabels)
axt.set_xlabel('Ionized Fraction')

fig.tight_layout(pad=0.5)
lines = l1+l2+l3
labels = [l.get_label() for l in lines]
ax1.legend(handles=lines, labels=labels)
fig.savefig('{:s}/plots_v2/model_stats2.pdf'.format(root_dir))
