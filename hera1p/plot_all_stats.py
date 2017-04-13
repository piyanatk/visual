import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
group = ['binning', 'windowing']
stat = ['var', 'skew', 'kurt']
nstat = len(stat)
ngroup = len(group)
nbandwidth = len(bandwidth)
ntelescope = len(telescope)
signal_field = 25
noise_nfield = 20
if noise_nfield < 200:
    idx = np.random.randint(0, 200, noise_nfield)
else:
    idx = np.arange(200)


# Plotting properties
l1 = 'k'  # signal
l2 = 'r'  # model
l3 = 'b'  # average
# Orange
# s1 = '#fff5eb'
# s2 = '#fdd49e'
# Blue
s1 = '#deebf7'
s2 = '#9ecae1'
# Green
# s1 = '#f7fcf5'
# s2 = '#c7e9c0'

ls1 = '-'
ls2 = '--'
ls3 = (8, (5, 1, 1, 1))

tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
handles = [Line2D([], [], c=l1, ls=ls1), Line2D([], [], c=l3, ls=ls3),
           Line2D([], [], c=l2, ls=ls2),
           Patch(fc=s2, alpha=1),
           Patch(fc=s1, alpha=1)]
labels = ['Field {:d}'.format(signal_field),
          '{:d}-Field Averaged'.format(noise_nfield),
          'Full-sky Model',
          'Sample Variance Error'.format(noise_nfield),
          'Sample Variance\n& Thermal Noise Errors']
ylabels = ['Variance', 'Skewness', 'Kurtosis']
# Collect data and plot
for j in range(ngroup):
    g = group[j]
    for k in range(ntelescope):
        t = telescope[k]
        for l in range(nbandwidth):
            b = bandwidth[l]
            plt.close()
            fig, ax = plt.subplots(3, 1, sharex='all', figsize=(4.25, 4))
            for i in range(nstat):
                s = stat[i]
                pn = pd.read_hdf(
                    '{:s}/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                        .format(data_dir, t, t, b, g)
                )
                hpx = pd.read_hdf(
                    '{:s}/healpix/{:s}/{:s}_hpx_interp_21cm_cube_l128_stats_df_bw{:.2f}MHz_{:s}.h5'
                        .format(data_dir, t, t, b, g)
                )
                model = hpx[s]
                signal = pn.iloc[signal_field][s]
                mean_signal = pn[idx].mean(axis=0)[s]
                sample_err = pn[idx].std(axis=0)[s]
                noise_err = pn[idx].mean(axis=0)[s+'_err']
                combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                x = mean_signal.index
                ax[i].plot(x, signal, c=l1, ls=ls1)
                ax[i].plot(x, model, c=l2, ls=ls2)
                ax[i].plot(x, mean_signal, c=l3, ls=ls3)
                ax[i].fill_between(x, mean_signal - combine_err,
                                   mean_signal + combine_err,
                                   color=s1, alpha=1)
                # ax[i].fill_between(x, mean_signal - noise_err,
                #                    mean_signal + noise_err,
                #                    color=s2, alpha=1)
                ax[i].fill_between(x, mean_signal - sample_err,
                                   mean_signal + sample_err,
                                   color=s2, alpha=1)
                if s == 'var':
                    ax[i].set_ylim(0, np.max((mean_signal.max(), signal.max())) * 1.2)
                else:
                    ax[i].set_ylim(np.min((mean_signal.min(), signal.min())) * 1.2,
                                   np.max((mean_signal.max(), signal.max())) * 1.2)
                ax[i].set_xlim(x.min(), x.max())
                ax[i].set_ylabel(ylabels[i])
                # ax[i].grid('on')
                if s != 'var':
                    ax[i].axhline(0, ls='--', c='k')
            ax[2].set_xlabel('Frequency [MHz]')
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.89,
                                hspace=0)
            # fig.suptitle('{:s} {:.2f} MHz {:s} ({:d} Fields Errors)'
            #              .format(tlabels[k], b, g.capitalize(), noise_nfield),
            #              fontsize='large')
            fig.legend(handles=handles, labels=labels, loc='upper center',
                       ncol=2, fontsize='small', handlelength=2.7)
            outdir = '/Users/piyanat/Google/research/hera1p/plots/all_stats/' \
                     '{:s}/field{:d}'.format(t, signal_field)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fig.savefig(
                '{:s}/all_stats_{:s}_field{:d}_averaged{:d}field'
                '_bw{:.2f}MHz_{:s}.pdf'
                .format(outdir, t, signal_field, noise_nfield, b, g)
            )
