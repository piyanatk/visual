import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats/healpix'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
group = ['binning', 'windowing']
stat = ['var', 'skew', 'kurt']
nstat = len(stat)
ngroup = len(group)
nbandwidth = len(bandwidth)
ntelescope = len(telescope)

# Plot parameters
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
# linestyle = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
linestyle = ['-', ':', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
marker = ['None', 'None', '.', 'x', '*', 'o', 's', '^', 'D']
color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61',
         '#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']
legend_handles = [Line2D([], [], ls=ls, color=c, marker=m)
                  for ls, c, m in zip(linestyle, color, marker)]
legend_labels = ['{:.2f} MHz'.format(bw) for bw in bandwidth]
# ylims = [[(0, 8), (), ()],
#          [(), (), ()],
#          [(0, 25), (0, 8), (0, 6)],
#          [(0, 25), (0, 8), (0, 6)],
#          [(0, 25), (0, 8), (0, 6)],
#          [(0, 28), (0, 8), (0, 6)],
#          [(0, 28), (0, 8), (0, 6)],
#          [(0, 30), (0, 15), (0, 8)],
#          [(0, 30), (0, 15), (0, 10)]]
# Collect data and plot
for k in range(ntelescope):
    t = telescope[k]
    plt.close()
    for p in range(ngroup):
        g = group[p]
        fig, ax = plt.subplots(3, 1, sharex='all', figsize=(5, 6))
        for l in range(nbandwidth):
            b = bandwidth[l]
            df = pd.read_hdf(
                '{:s}/{:s}/{:s}_hpx_interp_21cm_cube_l128_stats_df_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            for i in range(nstat):
                s = stat[i]
                signal = df[s]
                x = signal.index
                ax[i].plot(x, signal, ls=linestyle[l], marker=marker[l],
                           color=color[l])
        ax[1].axhline(0, c='k', ls='--')
        ax[2].axhline(0, c='k', ls='--')
        ax[0].set_xlim(140, 195)
        ax[0].set_ylabel('Variance [mK$^2$]')
        ax[1].set_ylabel('Skewness')
        ax[2].set_ylabel('Kurtosis')
        # fig.text(0.5, 0.98, '{:s} Statistics and SNR - Frequency {:s}'
        #          .format(tlabels[k], g.capitalize()),
        #          va='top', ha='center', fontsize='large')
        fig.text(0.5, 0.01, 'Frequency [MHz]', va='bottom', ha='center')
        fig.subplots_adjust(bottom=0.08, top=0.93, left=0.13, right=0.97,
                            hspace=0.08)
        fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=5, fontsize='x-small')
        fig.savefig('/Users/piyanat/Google/research/hera1p/plots/healpix_stats_bw/'
                    'healpix_stats_bw_{:s}_{:s}.pdf'.format(t, g))
        # plt.show()
