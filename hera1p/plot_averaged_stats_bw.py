import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
# bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0]
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
# color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61',
#          '#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']
# color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
color = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue
legend_handles = [Line2D([], [], ls=ls, color=c, marker=m)
                  for ls, c, m in zip(linestyle, color, marker)]
legend_labels = ['{:.2f} MHz'.format(bw) for bw in bandwidth]
ylims = [(0.05, 0.9), (-1, 1.5), (-1, 2.5)]

# x-axis coordinates for plotting and axes transformations.
fi, zi, xi = np.genfromtxt(
    '/Users/piyanat/Google/research/hera1p/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True)


# Collect data and plot
for k in range(ntelescope):
    t = telescope[k]
    plt.close()
    fig, ax = plt.subplots(3, 2, sharex='all', sharey='row', figsize=(8.5, 5.5))
    for p in range(ngroup):
        g = group[p]
        for l in range(nbandwidth):
            b = bandwidth[l]
            pn = pd.read_hdf(
                '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            for i in range(nstat):
                s = stat[i]
                signal = pn[idx].mean(axis=0)[s]
                # sample_err = pn[idx].std(axis=0)[s]
                # noise_err = pn[idx].mean(axis=0)[s+'_err']
                # combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                x = signal.index
                # snr = np.abs(signal)/combine_err
                ax[i, p].plot(x, signal, ls=linestyle[l], marker=marker[l],
                              color=color[l])
                if i > 0:
                    ax[i, p].axhline(0, c='k', ls='--', lw=1)
                ax[i, p].set_xlim(fi[0], fi[-1])
                # if s == 'skew':
                #     ax[i, p].set_ylim(-0.7, -0.3)
                    # fl, fb = fig.transFigure.inverted().transform(
                    #     ax[i, p].transData.transform([140, 0.1]))
                    # fr, ft = fig.transFigure.inverted().transform(
                    #     ax[i, p].transData.transform([180, 1.6]))
                    # sax = fig.add_axes([fl, fb, fr-fl, ft-fb])
                    # sax.set_xticklabels([])
                    # sax.set_yticklabels([])
                # if s == 'kurt':
                #     ax[i, p].set_ylim(0, 1)

    axt_xticklabels = np.arange(3, 10) * 0.1
    axt_xticklocs = np.interp(axt_xticklabels, xi, fi)

    axt0 = ax[0, 0].twiny()
    axt0.set_xlim(ax[0, 0].get_xlim())
    axt0.set_xticks(axt_xticklocs)
    axt0.set_xticklabels(axt_xticklabels)

    axt1 = ax[0, 1].twiny()
    axt1.set_xlim(ax[0, 1].get_xlim())
    axt1.set_xticks(axt_xticklocs)
    axt1.set_xticklabels(axt_xticklabels)

    # fig.text(0.5, 0.98, '{:s} Statistics and SNR - Frequency {:s}'
    #          .format(tlabels[k], g.capitalize()),
    #          va='top', ha='center', fontsize='large')
    fig.text(0.01, 0.75, 'Variance', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.5, 'Skewness', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.25, 'Kurtosis', rotation='vertical',
             ha='left', va='center')
    fig.text(0.5, 0.01, 'Observing Frequency [MHz]', va='bottom', ha='center')
    fig.text(0.5, 0.95, 'Ionized Fraction', va='top', ha='center')

    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='upper center', ncol=9, fontsize='x-small')

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.87,
                        hspace=0, wspace=0)

    fig.savefig(
        '/Users/piyanat/Google/research/hera1p/plots/averaged_stats_bw/'
        'averaged_stats_bw_{:s}.pdf'.format(t)
    )

    # plt.show()
