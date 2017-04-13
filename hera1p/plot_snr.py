import matplotlib
matplotlib.use('Agg')
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
c1 = '#08519c'
c2 = '#4292c6'
handles = [Line2D([], [], c=c1, ls=':'),
           Line2D([], [], c=c2, ls='-')]
labels = ['Sample Variance Error', 'Sample Variance & Thermal Noise Errors']

ylabels = ['Variance SNR', 'Skewness SNR', 'Kurtosis SNR']

# Collect data and plot
for j in range(ngroup):
    g = group[j]
    for k in range(ntelescope):
        b = bandwidth[k]
        for l in range(nbandwidth):
            t = telescope[l]
            plt.close()
            fig, ax = plt.subplots(3, 1, sharex='all', figsize=(5, 6.2))
            for i in range(nstat):
                s = stat[i]
                pn = pd.read_hdf(
                    '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                    .format(data_dir, t, t, b, g)
                )
                signal = pn[idx].mean(axis=0)[s]
                sample_err = pn[idx].std(axis=0)[s]
                noise_err = pn[idx].mean(axis=0)[s+'_err']
                combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                x = signal.index
                snr_sample_err = np.abs(signal)/sample_err
                snr_combine_err = np.abs(signal)/combine_err
                ax[i].plot(x, snr_combine_err, ls='-', c=c2)
                # ax[i].plot(x, np.abs(signal)/noise_err, 'g--')
                ax[i].plot(x, snr_sample_err, ls=':', c=c1)
                if snr_combine_err.min() < 1:
                    ax[i].axhline(1, ls='--', color='k')
                # ax[i].axvline(x[np.where(combine_err == combine_err.min())])
                # ax[i].set_ylim(0, snr_combine_err.max()*1.2)
                ax[i].set_xlim(x.min(), x.max())
                ax[i].set_ylabel(ylabels[i])
            ax[2].set_xlabel('Frequency [MHz]')
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.89,
                                hspace=0.1)
            # fig.suptitle('SNR {:s} {:.2f} MHz {:s}'
            #              .format(t.upper(), b, g.capitalize()),
            #              fontsize='large')
            fig.legend(handles=handles, labels=labels, loc='upper center',
                       ncol=2, fontsize='small')
            fig.savefig(
                '/Users/piyanat/Google/research/hera1p/plots/snr/'
                '{:s}/snr_{:s}_{:.2f}MHz_{:s}.pdf'.format(t, t, b, g)
            )
