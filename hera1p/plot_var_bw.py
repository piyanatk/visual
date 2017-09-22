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
linestyle = ['-', ':', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
marker = ['None', 'None', '.', 'x', '*', 'o', 's', '^', 'D']
rcolor = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
bcolor = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue

# x-axis coordinates for plotting and axes transformations.
fi, zi, xi = np.genfromtxt(
    '/Users/piyanat/Google/research/hera1p/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True)


# Collect data and plot
for k in range(ntelescope):
    t = telescope[k]
    plt.close()
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    for p in range(ngroup):
        g = group[p]
        for l in range(nbandwidth):
            b = bandwidth[l]
            pn = pd.read_hdf(
                '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            signal = pn[idx].mean(axis=0)['var']
            x = signal.index
            if g == 'windowing':
                # c = rcolor[0]
                c = '0.5'
                m = '.'
                ls = 'None'
                # alpha = 0.2
                zorder = 1
            else:
                c = 'k'
                # c = bcolor[l]
                ls = linestyle[l]
                m = marker[l]
                # alpha = 1
                zorder = 2
            ax.plot(x, signal, ls=ls, marker=m, color=c, zorder=zorder)
            ax.set_xlim(fi[0], fi[-1])

    axt_xticklabels = np.arange(3, 10) * 0.1
    axt_xticklocs = np.interp(axt_xticklabels, xi, fi)

    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    axt.set_xticks(axt_xticklocs)
    axt.set_xticklabels(axt_xticklabels)

    ax.set_ylabel('Variance [mK]')
    ax.set_xlabel('Observing Frequency [MHz]')
    axt.set_xlabel('Ionized Fraction')

    # legend_handles = [Line2D([], [], ls='-', lw=3, color=rcolor[0])] + \
    #                  [Line2D([], [], ls=ls, color=c, marker=m)
    #                   for ls, c, m in zip(linestyle, bcolor, marker)]
    legend_handles = [Line2D([], [], ls='-', lw=3, color='0.5')] + \
                     [Line2D([], [], ls=ls, color='k', marker=m)
                      for ls, m in zip(linestyle, marker)]
    legend_labels = ['Windowing (All Bandwidths)'] + \
                    ['Binning {:.2f} MHz'.format(bw) for bw in bandwidth]
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc='upper center', ncol=3, fontsize='x-small')

    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.12, top=0.78)

    fig.savefig(
        '/Users/piyanat/Google/research/hera1p/plots/var_bw/'
        'var_bw_{:s}.pdf'.format(t)
    )

    # plt.show()
