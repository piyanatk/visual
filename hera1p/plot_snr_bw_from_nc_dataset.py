import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sim.utils import check_dir

# Data parameters
main_dir = '/Users/piyanat/Google/data/hera1p'
data_dir = '{:s}/stats'.format(main_dir)
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
# bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 8.0]
averaging = ['binning', 'windowing']
stat = ['var', 'skew', 'kurt']


# Plot parameters
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']
# linestyle = ['-', ':', '-', '-', '-', '-', '-', '-', '-']
linestyle = ['-', ':', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
marker = ['None', 'None', 'x', '.', '*', 'd']
# color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61',
#          '#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Red & blue
# color = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
color = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue
# color = ['k', 'k', 'k', 'k', 'k', 'k']  # Black
legend_handles = [Line2D([], [], ls=ls, color=c, marker=m)
                  for ls, c, m in zip(linestyle, color, marker)]
legend_labels = ['{:.2f} MHz'.format(bw) for bw in bandwidth]

# redshift coordinates for axes transformations.
fi, zi, xi = np.genfromtxt(
    '{:s}/interp_delta_21cm_f_z_xi.csv'.format(main_dir),
    delimiter=',', unpack=True)


# Make plot
for i in range(len(telescope)):
# for i in [8]:
    plt.close()
    fig, ax = plt.subplots(3, 2, sharex='all', sharey='row', figsize=(8.5, 5.5))
    lines = []
    for j in range(len(bandwidth)):
    # for j in [0, 1, 2, 3, 4, 5]:
        ds = xr.open_dataset('{:s}/hera1p_all_stats_bw{:.2f}MHz.nc'
                             .format(data_dir, bandwidth[j]))
        x = ds.frequency.values
        snr = ds['drift_scan_snr_quad_sum']
        for l in range(len(stat)):
            for m in range(len(averaging)):
                y = snr.isel(telescope=i, stat=l, averaging=m).values
                ll = ax[l, m].plot(x, y, marker=marker[j], ls=linestyle[j])
                ax[l, m].axhline(1, c='k', ls='--', lw=1)
        lines += ll
    ax[0, 0].set_xlim(fi[0], fi[-1])

    # Make upper x-axis with redshift coordinates
    axt_xticklabels = np.arange(3, 10) * 0.1
    axt_xticklocs = np.interp(axt_xticklabels, xi, fi)
    axt0 = ax[0, 0].twiny()
    axt0.set_xlim(ax[0, 0].get_xlim())
    axt0.set_xticks(axt_xticklocs)
    axt0.set_xticklabels(axt_xticklabels)
    axt1 = ax[0, 1].twiny()
    axt1.set_xlim(ax[0, 1].get_xlim())
    axt1.set_xticks(axt_xticklocs)
    axt1.set_xticklabels(axt_xticklabels)

    # Axes labels
    # fig.text(0.5, 0.98, '{:s} Statistics and SNR - Frequency {:s}'
    #          .format(tlabels[k], g.capitalize()),
    #          va='top', ha='center', fontsize='large')
    fig.text(0.01, 0.75, 'Variance SNR', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.5, 'Skewness SNR', rotation='vertical',
             ha='left', va='center')
    fig.text(0.01, 0.25, 'Kurtosis SNR', rotation='vertical',
             ha='left', va='center')
    fig.text(0.5, 0.01, 'Observed Frequency [MHz]', va='bottom', ha='center')
    fig.text(0.5, 0.95, 'Ionized Fraction', va='top', ha='center')

    # Legend
    fig.legend(handles=lines, labels=legend_labels,
               loc='upper center', ncol=9, fontsize='x-small')

    # Tidy up
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.87,
                        hspace=0, wspace=0)

    # Save out
    out_dir = '{:s}/plots_v2/snr_bw'.format(main_dir)
    check_dir(out_dir)
    fig.savefig('{:s}/snr_bw_{:s}.pdf'.format(out_dir, telescope[i]))

    # plt.show()
