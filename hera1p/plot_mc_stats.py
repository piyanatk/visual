import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


l1 = 'b'
l2 = 'r'
l3 = 'k'
s1 = '#ece2f0'
s2 = '#a6bddb'
s3 = '#1c9099'

ls1 = ':'
ls2 = '--'
ls3 = '-'


if __name__ == "__main__":
    # Data parameters
    data_dir = '/Users/piyanat/Google/research/hera1p/stats'
    telescope = ['hera19', 'hera37', 'hera61', 'hera91',
                 'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
    # bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    bandwidth = [0.08]
    # group = ['binning', 'windowing']
    group = ['binning']
    stat = ['var', 'skew', 'kurt']
    nstat = len(stat)
    ngroup = len(group)
    nbandwidth = len(bandwidth)
    ntelescope = len(telescope)
    signal_field = 0
    noise_nfield = 200
    if noise_nfield < 200:
        idx = np.random.randint(0, 200, noise_nfield)
    else:
        idx = np.arange(200)
    handles = [Line2D([], [], c=l3, ls=ls3), Line2D([], [], c=l1, ls=ls1, lw=2),
               Line2D([], [], c=l2, ls=ls2), Patch(fc=s1, alpha=1)]
    labels = ['Field {:d} Signal'.format(signal_field), 'Full-sky Model',
              '{:d} Field Averaged Signal'.format(noise_nfield),
              '{:d} Field Sample Error'.format(noise_nfield)]
    # Collect data and plot
    for j in range(ngroup):
        for k in range(ntelescope):
            for l in range(nbandwidth):
                plt.close()
                fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
                for i in range(nstat):
                    s = stat[i]
                    g = group[j]
                    b = bandwidth[l]
                    t = telescope[k]
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
                    ax[i].plot(x, model, c=l1, ls=ls1, lw=2)
                    ax[i].plot(x, mean_signal, c=l2, ls=ls2)
                    ax[i].plot(x, signal, c=l3, ls=ls3)
                    # ax[i].plot(x, mean_signal - sample_err, c=l1, ls=':')
                    # ax[i].plot(x, mean_signal + sample_err, c=l1, ls=':')
                    ax[i].fill_between(x, mean_signal - combine_err,
                                       mean_signal + combine_err,
                                       color=s3, alpha=1)
                    ax[i].fill_between(x, mean_signal - noise_err,
                                       mean_signal + noise_err,
                                       color=s2, alpha=1)
                    ax[i].fill_between(x, mean_signal - sample_err,
                                       mean_signal + sample_err,
                                       color=s1, alpha=1)
                    if s == 'var':
                        ax[i].set_ylim(0, (mean_signal + sample_err).max() * 1.25)
                    else:
                        ax[i].set_ylim((mean_signal - sample_err).min() * 1.25,
                                       (mean_signal + sample_err).max() * 1.25)
                    ax[i].set_xlim(x.min(), x.max())
                    ax[i].set_ylabel(s)
                    ax[i].grid('on')
                ax[2].set_xlabel('Frequency [MHz]')
                fig.subplots_adjust(top=0.87)
                fig.suptitle('{:s} {:.2f} MHz {:s} ({:d} Fields Errors)'
                             .format(t.upper(), b, g.capitalize(), noise_nfield),
                             fontsize='large')
                fig.legend(handles=handles, labels=labels, loc=(0.15, 0.89),
                           ncol=2)
                # fig.savefig(
                #     '/Users/piyanat/Google/research/hera1p/plots/mc_stats/'
                #     '{:s}_mc_stats_plot_field{:d}_noise{:d}field_bw{:.2f}MHz_{:s}.pdf'.format(t, signal_field, noise_nfield, b, g)
                # )
                plt.show()