import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
group = ['binning', 'windowing']
stat = ['var', 'skew', 'kurt']
nstat = len(stat)
ngroup = len(group)
nbandwidth = len(bandwidth)
ntelescope = len(telescope)
nfield = 20
if nfield < 200:
    idx = np.random.randint(0, 200, nfield)
else:
    idx = np.arange(200)

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
        fig, ax = plt.subplots(3, 2, sharex='all', figsize=(8.5, 4))
        snr_max = [0, 0, 0]
        for l in range(nbandwidth):
            b = bandwidth[l]
            pn = pd.read_hdf(
                '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            for i in range(nstat):
                s = stat[i]
                signal = pn[idx].mean(axis=0)[s]
                sample_err = pn[idx].std(axis=0)[s]
                noise_err = pn[idx].mean(axis=0)[s+'_err']
                combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                x = signal.index
                snr = np.abs(signal)/combine_err
                if snr.max() > snr_max[i]:
                    snr_max[i] = snr.max()
                ax[i, 0].plot(x, signal, ls=linestyle[l], marker=marker[l],
                              color=color[l])
                ax[i, 1].plot(x, snr, ls=linestyle[l], marker=marker[l],
                              color=color[l])
                # ax[i].axvline(x[np.where(combine_err == combine_err.min())])
                # if s == 'var':
                #     ax[i].set_ylim(0, (signal/combine_err).max()*2)
                # else:
                #     ax[i].set_ylim((signal/combine_err).min()*2,
                #                    (signal/combine_err).max()*2)
                # ax[i].set_ylabel(s)
                # ax[i, 1].set_ylim(*ylims[k, i])
        ax[1, 0].axhline(0, c='k', ls='--')
        ax[2, 0].axhline(0, c='k', ls='--')
        for i in range(3):
            if snr_max[i] > 1.0:
                ax[i, 1].axhline(1, c='k', ls='--')
        ax[0, 0].set_xlim(140, 195)
        ax[0, 0].set_ylabel('Variance [mK$^2$]')
        ax[1, 0].set_ylabel('Skewness')
        ax[2, 0].set_ylabel('Kurtosis')
        ax[0, 1].set_ylabel('Variance SNR')
        ax[1, 1].set_ylabel('Skewness SNR')
        ax[2, 1].set_ylabel('Kurtosis SNR')
        # fig.text(0.5, 0.98, '{:s} Statistics and SNR - Frequency {:s}'
        #          .format(tlabels[k], g.capitalize()),
        #          va='top', ha='center', fontsize='large')
        fig.text(0.5, 0.005, 'Frequency [MHz]', va='bottom', ha='center')
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.94,
                            hspace=0, wspace=0.16)
        fig.legend(handles=legend_handles, labels=legend_labels,
                   loc='upper center', ncol=9, fontsize='x-small')
        fig.savefig(
            '/Users/piyanat/Google/research/hera1p/plots/averaged_stats_snr_bw/'
            'averaged_stats_snr_bw_{:s}_{:s}.pdf'.format(t, g)
        )
        # plt.show()
