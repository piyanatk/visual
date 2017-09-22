import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sim.utils import check_dir


# Data parameters
main_dir = '/Users/piyanat/Google/data/hera1p'
data_dir = '{:s}/stats'.format(main_dir)
# telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
#              'hera169', 'hera217', 'hera271', 'hera331']
telescope = ['hera19', 'hera37', 'hera127', 'hera217', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
averaging = ['binning', 'windowing']

# Plot parameters
linestyles = [(15, (20, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
              (10, (15, 1, 1, 1, 1, 1, 1, 1)),
              (5, (10, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), '-']
# colors = ['#67000d', '#a50026', '#d73027', '#f46d43', '#fdae61']  # Red
colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#08306b']  # Blue
# colors = ['0.4', '0.3', '0.2', '0.1', '0.0']  # Gray
ylabels = [['Variance', 'Variance SNR'],
           ['Skewness', 'Skewness SNR'],
           ['Kurtosis', 'Kurtosis SNR']]
telid = [0, 1, 4, 6, 8]

# Legend
legend_handles = [Line2D([], [], c=cl, ls=l) for cl, l in zip(colors, linestyles)]
legend_labels = ['HERA19', 'HERA37', 'HERA128', 'HERA240 Core', 'HERA350 Core']

# redshift coordinates for axes transformations.
fi, zi, xi = np.genfromtxt(
    '{:s}/interp_delta_21cm_f_z_xi.csv'.format(main_dir),
    delimiter=',', unpack=True)


# Make plot
for i in range(len(bandwidth)):
    ds = xr.open_dataset('{:s}/hera1p_all_stats_bw{:.2f}MHz.nc'
                         .format(data_dir, bandwidth[i]))
    x = ds.frequency.values
    s = ds['drift_scan_stats']
    snr = ds['drift_scan_snr_quad_sum']
    for j in range(len(averaging)):
        plt.close()
        fig, ax = plt.subplots(3, 2, sharex='all', figsize=(8.5, 5.5))
        lines = []
        for k in range(5):
            for l in range(3):
                for m in range(2):
                    if m == 0:
                        y = s.isel(averaging=j, telescope=telid[k], stat=l)
                        if l != 0:
                            ax[l, m].axhline(0, c='k', ls='--', lw=0.8)
                    else:
                        y = snr.isel(averaging=j, telescope=telid[k], stat=l)
                        ax[l, m].axhline(1, c='k', ls='--', lw=0.8)

                    ll = ax[l, m].plot(x, y, ls=linestyles[k])
            lines += ll
        ax[0, 0].set_xlim(x[0], x[-1])

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

        fig.text(0.01, 0.75, 'Variance', rotation='vertical',
                 ha='left', va='center')
        fig.text(0.01, 0.5, 'Skewness', rotation='vertical',
                 ha='left', va='center')
        fig.text(0.01, 0.25, 'Kurtosis', rotation='vertical',
                 ha='left', va='center')
        fig.text(0.51, 0.75, 'Variance SNR', rotation='vertical',
                 ha='left', va='center')
        fig.text(0.51, 0.5, 'Skewness SNR', rotation='vertical',
                 ha='left', va='center')
        fig.text(0.51, 0.25, 'Kurtosis SNR', rotation='vertical',
                 ha='left', va='center')

        fig.text(0.5, 0.01, 'Observed Frequency [MHz]', va='bottom',
                 ha='center')
        fig.text(0.5, 0.95, 'Ionized Fraction', va='top', ha='center')

        # Legend
        fig.legend(handles=lines, labels=legend_labels,
                   loc='upper center', ncol=5,
                   handlelength=4, fontsize='x-small')

        # Tidy up
        fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.87,
                            hspace=0, wspace=0.2)

        # Save out
        out_dir = '{:s}/plots_v2/stats_snr_heraxx'.format(main_dir)
        check_dir(out_dir)
        fig.savefig('{:s}/stats_snr_heraxx_bw{:.2f}MHz_{:s}.pdf'
                    .format(out_dir, bandwidth[i], averaging[j]))

        # plt.show()
