import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


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
ls3 = (10, (7, 2, 1, 2))


# Data parameters
data_dir = '/Users/piyanat/Google/research/hera1p/stats'
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
# bandwidth = [0.08]
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
handles = [Line2D([], [], c=l3, ls=ls3), Line2D([], [], c=l2, ls=ls2),
           Patch(fc=s2, alpha=1), Patch(fc=s1, alpha=1)]
labels = ['{:d} Field Averaged Signal'.format(nfield), 'Full-sky Model',
          '{:d} Field Sample Error'.format(nfield),
          'Sample Error & Thermal Noise Error']
# Collect data and plot
for j in range(ngroup):
    g = group[j]
    for k in range(ntelescope):
        t = telescope[k]
        for l in range(nbandwidth):
            b = bandwidth[l]
            plt.close()
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
            for i in range(nstat):
                s = stat[i]
                pn = pd.read_hdf(
                    '{:s}/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                    .format(data_dir, t, t, b, g)
                )
                hpx = pd.read_hdf(
                    '{:s}/healpix/{:s}/{:s}_hpx_interp_21cm_cube'
                    '_l128_stats_df_bw{:.2f}MHz_{:s}.h5'
                    .format(data_dir, t, t, b, g)
                )
                model = hpx[s]
                signal = pn[idx].mean(axis=0)[s]
                sample_err = pn[idx].std(axis=0)[s]
                noise_err = pn[idx].mean(axis=0)[s+'_err']
                combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                x = signal.index
                ax[i].plot(x, signal, c=l3, ls=ls3)
                ax[i].plot(x, model, c=l2, ls=ls2)
                # ax[i].plot(x, signal-sample_err, c='0.5', ls='-')
                # ax[i].plot(x, signal+sample_err, c='0.5', ls='-')
                ax[i].fill_between(x, signal - combine_err,
                                   signal + combine_err,
                                   color=s1, alpha=1)
                # ax[i].fill_between(x, signal - noise_err,
                #                    signal + noise_err,
                #                    color=s2, alpha=1)
                ax[i].fill_between(x, signal-sample_err, signal+sample_err,
                                   color=s2, alpha=1)
                if s == 'var':
                    ax[i].set_ylim(0, (signal+sample_err).max()*1.25)
                else:
                    ax[i].set_ylim((signal-sample_err).min()*1.25,
                                   (signal+sample_err).max()*1.25)
                ax[i].set_xlim(x.min(), x.max())
                ax[i].set_ylabel(s)
                ax[i].grid('on')
            ax[2].set_xlabel('Frequency [MHz]')
            fig.subplots_adjust(top=0.85)
            fig.suptitle('{:s} {:.2f} MHz {:s} ({:d} Fields)'
                         .format(tlabels[k], b, g.capitalize(), nfield),
                         fontsize='large')
            fig.legend(handles=handles, labels=labels, loc=(0.1, 0.86),
                       ncol=2, fontsize='medium')
            fig.savefig(
                '/Users/piyanat/Google/research/hera1p/plots/averaged_stats/'
                '{:s}/averaged_stats_{:s}_{:d}field_{:.2f}MHz_{:s}.pdf'
                .format(t, t, nfield, b, g)
            )
