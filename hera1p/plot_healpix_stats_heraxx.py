import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

data_dir = '/Users/piyanat/Google/research/hera1p/stats/healpix'
# telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
#              'hera169', 'hera217', 'hera271', 'hera331']
# telescope = ['hera331']
telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
labels = ['HERA19', 'HERA37', 'HERA128', 'HERA240 Core', 'HERA350 Core']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
stats = ['var', 'skew', 'kurt']
group = ['binning', 'windowing']

# ls = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
linestyles = [(15, (20, 1, 1, 1, 1, 1, 1, 1, 1, 1)), (10, (15, 1, 1, 1, 1, 1, 1, 1)),
      (5, (10, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), '-']
m = ['None', 'None', 'None', 'x', '*', 'o', 's', '^', 'D']
colors = [
    '#67000d',
    '#a50026',
    '#d73027',
    '#f46d43',
    '#fdae61'
]
#     '#abd9e9',
#     '#74add1',
#     '#4575b4',
#     '#313695',
#     '#08306b'
# ]
ylabels1 = ['Variance', 'Skewness', 'Kurtosis']
# ylabel1 = ['$S_2$', '$S_3$', '$S_4$']
# ylabel2 = [r'$\frac{|S_2|}{\sigma_{n,S_2}}$',
#            r'$\frac{|S_3|}{\sigma_{n,S_3}}$',
#            r'$\frac{|S_4|}{\sigma_{n,S_4}}$']
# ylabel2 = ['$SNR_{S_2}$', '$SNR_{S_3}$', '$SNR_{S_4}$']
ylabels2 = ['Variance SNR', 'Skewness SNR', 'Kurtosis SNR']
ylim1 = [(0, 1), (-1, 1.5), (-1, 2.5)]
ylim2 = [(0, 20), (0, 16), (0, 10)]


for i in range(len(group)):
    g = group[i]
    for j in range(len(bandwidth)):
        b = bandwidth[j]
        plt.close()
        fig, ax = plt.subplots(len(stats), 1, sharex='all', figsize=(5, 6))
        for k in range(len(telescope)):
            t = telescope[k]
            df = pd.read_hdf(
                '{:s}/{:s}/{:s}_hpx_interp_21cm_cube_l128_stats_df_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            x = df.index
            for p in range(len(stats)):
                s = stats[p]
                ax[p].plot(x, df[s], ls=linestyles[k], c=colors[k])
                if s != 'var':
                    ax[p].axhline(0, c='k', ls='--')
                ax[p].set_ylim(*ylim1[p])
                ax[p].set_ylabel(ylabels1[p])
        ax[0].set_xlim(139, 195)
        # fig.suptitle('Full-sky Model Statistics by HERA Configurations.',
        #              size='large')
        handles = [Line2D([], [], c=cl, ls=l) for cl, l in zip(colors, linestyles)]
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1,
                            hspace=0.08, wspace=0.15)
        fig.text(0.5, 0.02, 'Frequency [MHz]', ha='center', va='bottom')
        fig.legend(handles=handles, labels=labels, ncol=3, loc='upper center',
                   handlelength=3.7, fontsize='small')
        fig.savefig('/Users/piyanat/Google/research/hera1p/plots/healpix_stats_heraxx/'
                    'healpix_stats_heraxx_bw{:.2f}MHz_{:s}.pdf'.format(b, g))