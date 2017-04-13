import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
# telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
#              'hera169', 'hera217', 'hera271', 'hera331']
# telescope = ['hera331']
telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
labels = ['HERA19', 'HERA37', 'HERA128', 'HERA240 Core', 'HERA350 Core']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
stats = ['var', 'skew', 'kurt']
group = ['binning', 'windowing']
nfield = 20
if nfield < 200:
    fid = np.random.randint(0, 200, nfield)
else:
    fid = np.arange(200)

# ls = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
linestyles = [(15, (20, 1, 1, 1, 1, 1, 1, 1, 1, 1)), (10, (15, 1, 1, 1, 1, 1, 1, 1)),
      (5, (10, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), '-']
m = ['None', 'None', 'None', 'x', '*', 'o', 's', '^', 'D']
# colors = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue
ylabels1 = ['Variance', 'Skewness', 'Kurtosis']
# ylabel1 = ['$S_2$', '$S_3$', '$S_4$']
# ylabel2 = [r'$\frac{|S_2|}{\sigma_{n,S_2}}$',
#            r'$\frac{|S_3|}{\sigma_{n,S_3}}$',
#            r'$\frac{|S_4|}{\sigma_{n,S_4}}$']
# ylabel2 = ['$SNR_{S_2}$', '$SNR_{S_3}$', '$SNR_{S_4}$']
ylabels2 = ['Variance SNR', 'Skewness SNR', 'Kurtosis SNR']
ylim1 = [(-0.1, 0.9), (-0.8, 1.8), (-0.8, 2)]
ylim2 = [(0, 20), (0, 9), (0, 7.5)]


for i in range(len(group)):
    g = group[i]
    for j in range(len(bandwidth)):
        b = bandwidth[j]
        plt.close()
        fig, ax = plt.subplots(len(stats), 2, sharex='all', figsize=(8.5, 4))
        for k in range(len(telescope)):
            t = telescope[k]
            pn = pd.read_hdf(
                '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            mean_stats = pn.iloc[fid].mean(axis=0)
            sample_err = pn.iloc[fid].std(axis=0)
            x = mean_stats.index
            for p in range(len(stats)):
                s = stats[p]
                ax[p, 0].plot(x, mean_stats[s], ls=linestyles[k], c=colors[k])
                if s != 'var':
                    ax[p, 0].axhline(0, c='k', ls='--')

                signal = np.abs(mean_stats[s])
                noise = np.sqrt(mean_stats[s+'_err'] ** 2 + sample_err[s] ** 2)
                snr = signal/noise
                ax[p, 1].plot(x, snr, ls=linestyles[k], c=colors[k])
                ax[p, 1].axhline(1, c='k', ls='--')
                # ax[p, 0].set_ylim(*ylim1[p])

                ax[p, 0].set_ylabel(ylabels1[p])
                ax[p, 1].set_ylabel(ylabels2[p])
                # ax[p, 1].set_yscale('log')
                # ax[p, 1].set_ylim(*ylim2[p])
        ax[0, 0].set_xlim(139, 195)
        handles = [Line2D([], [], c=cl, ls=l)
                   for cl, l in zip(colors, linestyles)]
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.93,
                            hspace=0.05, wspace=0.17)
        fig.text(0.5, 0.02, 'Frequency [MHz]', ha='center', va='bottom')
        fig.legend(handles=handles, labels=labels, ncol=9, loc='upper center',
                   handlelength=3.7, fontsize='small')
        fig.savefig(
            '/Users/piyanat/Google/research/hera1p/plots/averaged_stats_snr_heraxx/'
            'averaged_stats_snr_heraxx_bw{:.2f}MHz_{:s}.pdf'.format(b, g)
        )
