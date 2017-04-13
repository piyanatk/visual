import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


if __name__ == "__main__":
    plt.close()
    # Data parameters
    data_dir = '/Users/piyanat/Google/research/hera1p/mc_stats'
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
    if nfield < 200:
        idx = np.random.randint(0, 200, nfield)
    else:
        idx = np.arange(200)
    # Collect data
    arr = np.empty((nstat, ngroup, nbandwidth, ntelescope))
    for i in range(nstat):
        for j in range(ngroup):
            for k in range(ntelescope):
                for l in range(nbandwidth):
                    s = stat[i]
                    g = group[j]
                    b = bandwidth[k]
                    t = telescope[l]
                    pn = pd.read_hdf(
                        '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                        .format(data_dir, t, t, b, g)
                    )
                    signal = pn[idx].mean(axis=0)[s]
                    sample_err = pn[idx].std(axis=0)[s]
                    noise_err = pn[idx].mean(axis=0)[s+'_err']
                    noise = np.sqrt(sample_err ** 2 + noise_err ** 2)
                    arr[i, j, k, l] = -(signal / noise).min()
    fig = plt.figure(figsize=(8, 10))
    gs1 = GridSpec(nrows=3, ncols=2, width_ratios=(1, 1, 0.05),
                   hspace=0.05, wspace=0.05,
                   left=0.1, bottom=0.11, right=0.87, top=0.93)
    gs2 = GridSpec(nrows=3, ncols=1, hspace=0.05, wspace=0.05,
                   left=0.88, bottom=0.11, right=0.89, top=0.93)
    ticks = np.arange(9) + 0.5
    # title = ['Variance', '', 'kurt']
    ax = np.empty((3, 2), dtype=object)
    cax = np.empty(3, dtype=object)
    for i in range(3):
        vmin = arr[i, :, :, :].min()
        vmax = arr[i, :, :, :].max()
        for j in range(2):
            data = arr[i, j, :, :]
            ax[i, j] = fig.add_subplot(gs1[i, j])
            im = ax[i, j].pcolorfast(arr[i, j, :, :], vmin=vmin, vmax=vmax,
                                     cmap='Reds')
            if i == 2:
                ax[i, j].set_xticks(ticks)
                ax[i, j].set_xticklabels(telescope, rotation=-45, ha='left')
            else:
                ax[i, j].set_xticklabels([])
            if j == 0:
                ax[i, j].set_yticks(ticks)
                ax[i, j].set_yticklabels(bandwidth)
            else:
                ax[i, j].set_yticklabels([])
        cax = fig.add_subplot(gs2[i])
        fig.colorbar(im, cax=cax)
    fig.text(0.02, 0.55, 'Bin/Window Size [MHz]', rotation='vertical', va='center')
    fig.text(0.5, 0.02, 'Array Configurations', ha='center')
    fig.text(0.33, 0.94, 'Frequency Binning', ha='center')
    fig.text(0.66, 0.94, 'Frequency Windowing', ha='center')
    fig.text(0.97, 0.82, 'Variance', rotation='vertical', va='center', ha='center')
    fig.text(0.97, 0.55, 'Skewness', rotation='vertical', va='center', ha='center')
    fig.text(0.97, 0.25, 'Kurtosis', rotation='vertical', va='center', ha='center')
    fig.suptitle('Minimum SNR (Sample & Thermal Noise Errors) ({:d} Fields)'.format(nfield))
    fig.savefig('min_snr_{:d}_fields.pdf'.format(nfield))
