import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


c1 = '#1b9e77'
c2 = '#d95f02'
c3 = '#7570b3'


if __name__ == "__main__":
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
    if nfield < 200:
        idx = np.random.randint(0, 200, nfield)
    else:
        idx = np.arange(200)
    handles = [Line2D([], [], c=c1, ls='-'), Line2D([], [], c=c2, ls='--'),
               Line2D([], [], c=c3, ls=':')]
    labels = ['Combined Error', 'Thermal Noise Error',
              'Sample Error']
    # Collect data and plot
    for j in range(ngroup):
        for k in range(ntelescope):
            for l in range(nbandwidth):
                plt.close()
                fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
                for i in range(nstat):
                    s = stat[i]
                    g = group[j]
                    b = bandwidth[k]
                    t = telescope[l]
                    pn = pd.read_hdf(
                        '{:s}/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'
                        .format(data_dir, t, t, b, g)
                    )
                    sample_err = pn[idx].std(axis=0)[s]
                    noise_err = pn[idx].mean(axis=0)[s+'_err']
                    combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
                    x = sample_err.index
                    ax[i].plot(x, combine_err, ls='-', c=c1)
                    ax[i].plot(x, noise_err, ls='--', c=c2)
                    ax[i].plot(x, sample_err, ls=':', c=c3)
                    ax[i].axvline(x[np.where(combine_err == combine_err.min())])
                    ax[i].set_ylim(0, sample_err.max()*2)
                    ax[i].set_xlim(x.min(), x.max())
                    ax[i].set_ylabel(s)
                ax[2].set_xlabel('Frequency [MHz]')
                fig.subplots_adjust(top=0.9)
                fig.suptitle('SNR {:s} {:.2f} MHz {:s}'
                             .format(t.upper(), b, g.capitalize()),
                             fontsize='large')
                fig.legend(handles=handles, labels=labels, loc=(0.12, 0.92),
                           ncol=3, fontsize='medium')
                fig.savefig('/Users/piyanat/Google/research/hera1p/plots/error/'
                            'error_plot_{:s}_{:.2f}MHz_{:s}.pdf'.format(t, b, g))
                # plt.show()