import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


stats = ['var', 'skew', 'kurt']
telescopes = ['mwa128', 'hera37', 'hera331']
styles = ['k:', 'k--', 'k-']
stats_dir = '/Users/piyanat/research/pdf_paper/new_stats/'
pn = pd.Panel(
    dict(mwa128=pd.read_hdf(stats_dir + 'mwa128_gauss_stats_df_bw0.08MHz_windowing.h5'),
         hera37=pd.read_hdf(stats_dir + 'hera37_gauss_stats_df_bw0.08MHz_windowing.h5'),
         hera331=pd.read_hdf(stats_dir + 'hera331_gauss_stats_df_bw0.08MHz_windowing.h5'))
    )
xi = np.genfromtxt(
    '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', usecols=(2,), unpack=True
)
ylims = dict(var=(-0.1, 0.9), skew=(-1, 1.5), kurt=(-1, 2.5))
nticks = dict(var=6, skew=6, kurt=7)
ylabels = dict(var='Variance ($S_2$) [mK$^2$]',
               skew='Skewness ($S_3$)',
               kurt='Kurtosis ($S_4$)')

fig, axes = plt.subplots(nrows=3, sharex=False, sharey=False,
                         gridspec_kw=dict(hspace=0),
                         figsize=(8, 8))
axes_twin = []
colors = ['0.85', '0.70', '0.55']
for stat, ax in zip(stats, axes.ravel()):
    x = pn.major_axis
    for tel, st, cl in zip(telescopes, styles, colors):
        y = pn[tel][stat]
        yerr_max = y + pn[tel][stat + '_err']
        yerr_min = y - pn[tel][stat + '_err']
        ax.axhline(0, color='lightgray', linestyle='-')
        ax.fill_between(x, yerr_min, yerr_max, color=cl)
        ax.plot(x, y, st)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ylims[stat])
    ax.set_ylabel(ylabels[stat])
    ax.get_yaxis().set_label_coords(-0.07, 0.5)
    # ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))
    # nlocators = (abs(ylims[stat][0]) + abs(ylims[stat][1])) / 0.5
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks[stat], prune='upper'))

    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()

    # Add twin axes at the top
    axt = ax.twiny()
    axt.set_xlim(xi[0], xi[-1])
    axt.spines['bottom'].set_visible(False)
    axt.spines['left'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axt.xaxis.set_ticks_position('top')
    axt.minorticks_on()
    axes_twin.append(axt)

# Legend
handlers = [
    Line2D([], [], linestyle=':', color='black', linewidth=2),
    Line2D([], [], linestyle='--', color='black', linewidth=2),
    Line2D([], [], linestyle='-', color='black', linewidth=2),
    Patch(color=colors[0]),
    Patch(color=colors[1]),
    Patch(color=colors[2])
]
labels = [
    'MWA Phase I Core', 'HERA37', 'HERA331',
    'MWA Phase I Core Error', 'HERA37 Error', 'HERA331 Error'
]
axes[0].legend(handles=handlers, labels=labels, loc='upper left', ncol=1,
               fontsize='smaller')

# Tidy up
axes[2].set_xlabel('Frequency [MHz]')
axes_twin[1].xaxis.set_ticklabels([])
axes_twin[2].xaxis.set_ticklabels([])
axes_twin[0].set_xlabel('Ionized Fraction')
fig.tight_layout(rect=[0, 0, 0.99, 0.99])
fig.savefig(stats_dir + 'stats_thermal.pdf', dpi=200)
plt.close()
