import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
# telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
#              'hera169', 'hera217', 'hera271', 'hera331']
telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
stats = ['var', 'skew', 'kurt']
group = ['binning', 'windowing']
nfield = 200
if nfield < 200:
    fid = np.random.randint(0, 200, nfield)
else:
    fid = np.arange(200)

# Plot parameters
# ls = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
linestyles = [(15, (20, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
              (10, (15, 1, 1, 1, 1, 1, 1, 1)),
              (5, (10, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), '-']
# markers = ['None', 'None', 'None', 'x', '*', 'o', 's', '^', 'D']
# colors = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue
handles = [Line2D([], [], c=cl, ls=l) for cl, l in zip(colors, linestyles)]
labels = ['HERA19', 'HERA37', 'HERA128', 'HERA240 Core', 'HERA350 Core']

# x-axis coordinates for plotting and axes transformations.
fi, zi, xi = np.genfromtxt(
    '/Users/piyanat/Google/research/hera1p/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True)

for j in range(len(bandwidth)):
    b = bandwidth[j]
    plt.close()
    fig, ax = plt.subplots(1, 2, sharex='all', sharey='all',
                           figsize=(8.5, 5.5))
    for i in range(len(group)):
        g = group[i]
        for k in range(len(telescope)):
            t = telescope[k]
            pn = pd.read_hdf(
                '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                .format(data_dir, t, t, b, g)
            )
            # mean_stats = pn.iloc[fid].mean(axis=0)['kurt']
            sample_err = pn.iloc[fid].std(axis=0)['kurt']
            x = sample_err.index
            # signal = np.abs(mean_stats[s])
            # noise = np.sqrt(mean_stats[s+'_err'] ** 2 + sample_err[s] ** 2)
            # snr = np.abs(signal)/noise
            ax[i].plot(x, sample_err, ls=linestyles[k], c=colors[k])
    ax[0].set_xlim(fi[0], fi[-1])

    axt_xticklabels = np.arange(3, 10) * 0.1
    axt_xticklocs = np.interp(axt_xticklabels, xi, fi)

    axt0 = ax[0].twiny()
    axt0.set_xlim(ax[0].get_xlim())
    axt0.set_xticks(axt_xticklocs)
    axt0.set_xticklabels(axt_xticklabels)

    axt1 = ax[1].twiny()
    axt1.set_xlim(ax[1].get_xlim())
    axt1.set_xticks(axt_xticklocs)
    axt1.set_xticklabels(axt_xticklabels)

    fig.text(0.01, 0.75, 'Variance', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.5, 'Skewness', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.25, 'Kurtosis', rotation='vertical',
             ha='left', va='center')
    fig.text(0.5, 0.01, 'Observing Frequency [MHz]', va='bottom', ha='center')
    fig.text(0.5, 0.95, 'Ionized Fraction', va='top', ha='center')

    fig.legend(handles=handles, labels=labels, ncol=9, loc='upper center',
               handlelength=5.5, fontsize='x-small')

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.87,
                        hspace=0, wspace=0)

    fig.savefig(
        '/Users/piyanat/Google/research/hera1p/plots/'
        'sample_errors_kurtosis_heraxx/'
        'sample_errors_kurtosis_heraxx_bw{:.2f}MHz.pdf'.format(b)
    )

    # plt.show()
