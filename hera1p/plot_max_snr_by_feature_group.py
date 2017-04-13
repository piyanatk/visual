import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def get_fid(nf):
    if nf < 200:
        idx = np.random.randint(0, 200, nf)
    else:
        idx = np.arange(200)
    return idx


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
nfield = 200
nrun = 1

# Plot parameters
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
# marker = ['*', 'o', 's', '^', 'v', 'P', 'D', 'p', 'h']
# color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61',
#          '#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']
# legend_handles = [Line2D([], [], ls='None', marker=m, mec='k', mfc='none', ms=10)
#                   for m in marker]
# legend_handles.append(Line2D([], [], ls='None', marker='None'))
# legend_labels = ['{:.2f} MHz'.format(bw) for bw in bandwidth]
# legend_labels.append('Filled Marker indicates Binning')

# # Collect data
# snr_max = np.zeros((nrun, nstat, 3, ntelescope), dtype=float)
# bin_id = np.zeros((nrun, nstat, 3, ntelescope), dtype=int)
# bin_type = np.zeros((nrun, nstat, 3, ntelescope), dtype=int)
#
# for l in range(nrun):
#     idx = get_fid(nfield)
#     for i in range(ntelescope):
#         tl = telescope[i]
#         for j in range(nbandwidth):
#             bw = bandwidth[j]
#             for k in range(ngroup):
#                 gr = group[k]
#                 pn = pd.read_hdf(
#                     '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
#                     .format(data_dir, tl, tl, bw, gr)
#                 )
#                 for p in range(nstat):
#                     st = stat[p]
#                     signal = pn[idx].mean(axis=0)[st]
#                     sample_err = pn[idx].std(axis=0)[st]
#                     noise_err = pn[idx].mean(axis=0)[st + '_err']
#                     combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
#                     x = signal.index.values
#                     snr = (np.abs(signal)/combine_err).values
#                     temp = [np.max(snr[np.where(x < 180)]),
#                             np.max(snr[np.where((x >= 180) & (x < 190))]),
#                             np.max(snr[np.where((x >= 190))])]
#                     for q in range(3):
#                         if snr_max[l, p, q, i] < temp[q]:
#                             snr_max[l, p, q, i] = temp[q]
#                             bin_id[l, p, q, i] = j
#                             bin_type[l, p, q, i] = k
# snr_max = snr_max.mean(axis=0)
# bin_id = bin_id.mean(axis=0).astype(int)
# bin_type = bin_type.mean(axis=0).astype(int)
#
# # Plot
snr_arr, bw_arr, grp_arr = np.load(
    '/Users/piyanat/Google/research/hera1p/max_snr_{:d}fields_{:d}runs.npy'
    .format(nfield, nrun))
plt.close()
fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
axt = np.empty((3, 3), dtype=object)
for i in range(nstat):
    for j in range(3):
        x = range(ntelescope)
        y1 = []
        y2 = []
        facecolor = []
        for k in range(ntelescope):
            if j == 0:
                max_snr = snr_arr[k, i, :-2].max()
            elif j == 1:
                max_snr = snr_arr[k, i, -2]
            else:
                max_snr = snr_arr[k, i, -1]
            idx = np.where(snr_arr[k, i, :] == max_snr)[0][0]
            bw = int(bw_arr[k, i, :][idx])
            grp = grp_arr[k, i, :][idx]
            # m = marker[bw]
            if grp == 1:
                fc = 'w'
            else:
                fc = 'k'
            y1.append(max_snr)
            y2.append(bandwidth[bw])
            facecolor.append(fc)
        # ax[i, j].plot(k, max_snr, ls='none',
        #               marker=m, mfc=fc, mec='k', ms=10)
        ax[i, j].plot(x, y1, 'k-', zorder=1)
        axt[i, j] = ax[i, j].twinx()
        axt[i, j].plot(x, y2, 'k:', zorder=2)
        for xi, yi, fc in zip(x, y2, facecolor):
            axt[i, j].scatter(xi, yi, marker='s', facecolor=fc, edgecolor='k',
                              s=50, zorder=3)
        axt[i, j].set_ylim(0, 8.5)
ax[0, 0].set_ylabel('Variance')
ax[1, 0].set_ylabel('Skewness')
ax[2, 0].set_ylabel('Kurtosis')
axt[0, 2].set_ylabel(r'$\Delta \nu_{Max SNR}$ [MHz]')
axt[1, 2].set_ylabel(r'$\Delta \nu_{Max SNR}$ [MHz]')
axt[2, 2].set_ylabel(r'$\Delta \nu_{Max SNR}$ [MHz]')
ax[0, 0].set_xlim(-0.5, 8.5)
ax[0, 0].set_title(r'$\nu < 180\,MHz$')
ax[0, 1].set_title(r'$180\,MHz < \nu <= 190\,MHz$')
ax[0, 2].set_title(r'$\nu >=190\,MHz$')
for i in range(3):
    ax[2, i].xaxis.set_ticks(range(0, 9))
    ax[2, i].xaxis.set_ticklabels(telescope, rotation=-45)
# fig.text(0.5, 0.98, '{:s} Statistics'.format(tlabels[k]),
#          va='top', ha='center', fontsize='large')
# fig.text(0.5, 0.06, 'Frequency [MHz]', va='bottom', ha='center')
fig.subplots_adjust(bottom=0.08, top=0.92, left=0.08, right=0.92,
                    wspace=0.3)
handles = [Line2D([], [], c='k', ls='-'),
           Line2D([], [], c='k', ls=':', marker='s', mfc='k', mec='k'),
           Line2D([], [], c='k', ls=':', marker='s', mfc='w', mec='k')]
labels = ['Max SNR', r'$\Delta \nu_{Max SNR}$ (Binning)', r'$\Delta \nu_{Max SNR}$ (Windowing)']
fig.legend(handles=handles, labels=labels,
           loc='upper center', ncol=3, fontsize='medium', numpoints=1)
fig.savefig('/Users/piyanat/Google/research/hera1p/plots/'
            'max_snr_by_group_{:d}fields_{:d}runs.pdf'.format(nfield, nrun))
# plt.show()




