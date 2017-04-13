import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sim.settings import MWA_FREQ_EOR_ALL_80KHZ as freq
from sim.utils import bin_freqs


def get_bin(bin_width):
    native_ch_width = 0.08
    nchannel = int(np.ceil(bin_width / native_ch_width))
    channel_list = np.array_split(
        np.arange(705), np.arange(705, 0, -nchannel)[::-1]
    )[:-1]
    return channel_list


def get_fid(nf):
    if nf < 200:
        idx = np.random.randint(0, 200, nf)
    else:
        idx = np.arange(200)
    return idx


plt.close()
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
nfreq = len(freq)
nfield = 200
nruns = 1


# Collect data
arr = np.zeros((nruns, ntelescope, nstat, nbandwidth, ngroup, 705))
for h in range(nruns):
    idx = get_fid(nfield)
    for i in range(ntelescope):
        for j in range(nbandwidth):
            for k in range(ngroup):
                tl = telescope[i]
                bw = bandwidth[j]
                gr = group[k]
                pn = pd.read_hdf(
                    '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                    .format(data_dir, tl, tl, bw, gr)
                )
                for l in range(nstat):
                    st = stat[l]
                    signal = pn[idx].mean(axis=0)[st]
                    sample_err = pn[idx].std(axis=0)[st]
                    noise_err = pn[idx].mean(axis=0)[st + '_err']
                    combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                    snr = (np.abs(signal) / combine_err).values
                    if bw == 0.08:
                        arr[h, i, l, j, k, :] = snr
                    else:
                        chls = get_bin(bw)
                        for m in range(len(chls)):
                            bn = chls[m]
                            arr[h, i, l, j, k, bn] = snr[m]
np.save('/Users/piyanat/Google/research/hera1p/snr_{:d}fields_{:d}runs.npy'.format(nfield, nruns), arr)
arr = arr.mean(axis=0)

# Find maximum and rearrange data
chls = get_bin(8.0)
nbin = len(chls)
snr_arr = np.zeros((ntelescope, nstat, nbin))
bw_arr = np.zeros((ntelescope, nstat, nbin))
grp_arr = np.zeros((ntelescope, nstat, nbin), dtype=object)
for p in range(ntelescope):
    for q in range(nstat):
        for r in range(nbandwidth):
            for s in range(ngroup):
                for t in range(nbin):
                    bn = chls[t]
                    data_cut = arr[p, q, r, s, bn]
                    max_val = data_cut.max()
                    if snr_arr[p, q, t] <= max_val:
                        snr_arr[p, q, t] = max_val
                        bw_arr[p, q, t] = bandwidth[r]
                        grp_arr[p, q, t] = s
np.save('/Users/piyanat/Google/research/hera1p/max_snr_{:d}fields_{:d}runs.npy'.format(nfield, nruns),
        np.stack((snr_arr, bw_arr, grp_arr)))


# Plot
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
slabels = ['Variance', 'Skewness', 'Kurtosis']
x = bin_freqs(8.0)[1]
for u in range(ntelescope):
    for v in range(nstat):
        plt.close()
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(8, 10))
        max_snr = snr_arr[u, v, :]
        max_bw = bw_arr[u, v, :]
        bidx = (grp_arr[u, v, :] == 0)
        widx = (grp_arr[u, v, :] == 1)
        ax[0].plot(x[1:], max_snr[1:], 'k-')

        ax[1].plot(x[1:], max_bw[1:], 'k:', zorder=1)
        ax[1].scatter(x[bidx], max_bw[bidx], marker='s',
                      edgecolor='k', facecolor='k', zorder=2,
                      label='Binning', s=50)
        ax[1].scatter(x[widx], max_bw[widx], marker='s',
                      edgecolor='k', facecolor='w', zorder=2,
                      label='Windowing', s=50)
        ax[1].set_xlim(140, 195)
        ax[1].set_ylim(0, 8.5)
        ax[1].legend()
        ax[0].set_title('MAX SNR - {:s} {:s}'.format(tlabels[u], slabels[v]))
        ax[0].set_ylabel('MAX SNR ({:s})'.format(slabels[v]))
        ax[1].set_ylabel('Bandwidth with MAX SNR [MHz]')
        ax[1].set_xlabel('Observed Frequency [MHz] (8 MHz Block)')
        fig.savefig('/Users/piyanat/Google/research/hera1p/plots/'
                    'max_snr_8MHz_block/max_snr_8MHz_block_{:s}_{:s}_{:d}fields_{:d}runs.pdf'
                    .format(telescope[u], stat[v], nfield, nruns))
