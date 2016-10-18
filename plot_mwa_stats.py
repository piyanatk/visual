import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


stats_dir = '/Users/piyanat/research/pdf_paper/new_stats/'
pn = pd.Panel(
    dict(fhd=pd.read_hdf(stats_dir + 'mwa128_fhd_stats_df_bw0.08MHz_windowing.h5'),
         gauss=pd.read_hdf(stats_dir + 'mwa128_gauss_stats_df_bw0.08MHz_windowing.h5'),
         res=pd.read_hdf(stats_dir + 'mwa128_res_stats_df_bw0.08MHz_windowing.h5'))
)

f, z, xi = np.genfromtxt(
    '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True
)

stats = ['var', 'skew', 'kurt']
telescopes = ['fhd', 'gauss', 'res']
styles = ['k-', 'k--', 'k:']
ylims = dict(var=(0.0, 0.4), skew=(-1.0, 1.0), kurt=(-0.5, 1.5))
nticks = dict(var=8, skew=8, kurt=8)
# ylabels = dict(var='Variance [mK$^2$]', skew='Skewness', kurt='Kurtosis')
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
        # yerr_max = y + pn[tel][stat + '_err']
        # yerr_min = y - pn[tel][stat + '_err']
        # ax.axhline(0, color='lightgray', linestyle='-')
        # ax.fill_between(x, yerr_min, yerr_max, color=cl)
        ax.plot(x, y, st)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ylims[stat])
    ax.set_ylabel(ylabels[stat])
    ax.get_yaxis().set_label_coords(-0.09, 0.5)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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

    # Print correlation coefficient
    corr_coef = pn['fhd'][stat].corr(pn['gauss'][stat])
    text = 'PCC = {:.3f}'.format(corr_coef)
    ax.text(0.02, 0.9, text, transform=ax.transAxes)

    # Print soem 

# Legend
handlers = [
    Line2D([], [], linestyle='-', color='black'),
    Line2D([], [], linestyle='--', color='black'),
    Line2D([], [], linestyle=':', color='black'),
]
labels = ['FHD', 'Gaussian', 'Residual']
axes[0].legend(handles=handlers, labels=labels, loc=(0.02, 0.5),
               fontsize='medium', bbox_transform=axes[0].transAxes)

# Tidy up
axes[2].set_xlabel('Frequency [MHz]')
axes_twin[1].xaxis.set_ticklabels([])
axes_twin[2].xaxis.set_ticklabels([])
axes_twin[0].set_xlabel('Ionized Fraction')
fig.tight_layout(rect=[0, 0, 0.99, 0.99])
fig.savefig(stats_dir + 'stats_mwa.pdf', dpi=200)
plt.close()
