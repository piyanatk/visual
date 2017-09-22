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
l1 = ax1.plot(fi, df['var'], '-')
l2 = ax1.plot(fi, df['skew'], '--')
l3 = ax1.plot(fi, df['kurt'], '-.')
ax1.set_xlim(fi[0], fi[-1])
ax1.set_xlabel('Observed Frequency [MHz]')
ax1.set_ylabel('Statistical Values')
ax2 = ax1.twinx()
l4 = ax2.plot(fi, xi, ':')
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
handles = l1+l2+l3+l4
labels = ['$S_2$', '$S_3$', '$S_4$', '$x_i$']
ax1.legend(handles=handles, labels=labels, loc=(0.3, 0.45))
fig.savefig('{:s}/plots_v2/model_stats.pdf'.format(root_dir))
