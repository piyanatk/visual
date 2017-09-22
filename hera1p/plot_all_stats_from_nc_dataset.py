import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator

from sim.utils import check_dir


# Data parameters
main_dir = '/Users/piyanat/Google/data/hera1p'
data_dir = '{:s}/stats'.format(main_dir)
averaging = ['binning', 'windowing']
stat = ['var', 'skew', 'kurt']
telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
signal_field = 25

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
# labels = ['Field {:d}'.format(signal_field),
#           'Drift Scan', 'Full-sky',
#           'Sample Variance Error',
#           'Sample Variance\n& Thermal Noise Errors']

# redshift coordinates for axes transformations.
fi, zi, xi = np.genfromtxt(
    '{:s}/interp_delta_21cm_f_z_xi.csv'.format(main_dir),
    delimiter=',', unpack=True)


# for i in range(len(bandwidth)):
for i in [0]:
    ds = xr.open_dataset('{:s}/hera1p_all_stats_bw{:.2f}MHz.nc'
                         .format(data_dir, bandwidth[i]))
    x = ds.frequency.values
    sky_s = ds['full_sky_stats']
    field_s = ds['single_field_stats'].isel(field=signal_field)
    drift_scan_s = ds['drift_scan_stats']
    drift_scan_te_min = ds['drift_scan_stats'] - ds['drift_scan_thermal_errors']
    drift_scan_te_max = ds['drift_scan_stats'] + ds['drift_scan_thermal_errors']
    drift_scan_se_min = ds['drift_scan_stats'] - ds['drift_scan_sample_errors']
    drift_scan_se_max = ds['drift_scan_stats'] + ds['drift_scan_sample_errors']
    drift_scan_qse_min = ds['drift_scan_stats'] - ds['drift_scan_quad_sum_errors']
    drift_scan_qse_max = ds['drift_scan_stats'] + ds['drift_scan_quad_sum_errors']
    # for j in range(len(telescope)):
    for j in [8]:
        # for k in range(len(averaging)):
        for k in [1]:
            plt.close()
            fig, ax = plt.subplots(3, 1, sharex='all', figsize=(5, 5.5),
                                   gridspec_kw=dict(hspace=0))
            for l in range(3):
                selection = dict(telescope=j, averaging=k, stat=l)
                l1 = ax[l].plot(
                    x, field_s.isel(**selection),
                    label='Field {:d}'.format(signal_field),
                    ls=':', alpha=1, zorder=5
                )
                l2 = ax[l].plot(
                    x, sky_s.isel(**selection), label='Full sky',
                    ls='--', alpha=1, zorder=4
                )
                l3 = ax[l].plot(
                    x, drift_scan_s.isel(**selection), label='Drift scan',
                    ls='-', alpha=1, zorder=3
                )
                # ax[l].fill_between(x, drift_scan_te_min.isel(**selection),
                #                    drift_scan_te_max.isel(**selection),
                #                    alpha=0.5, color='0.7')
                s1 = ax[l].fill_between(
                    x, drift_scan_qse_min.isel(**selection),
                    drift_scan_qse_max.isel(**selection),
                    label='Sample variance &\nthermal noise uncertainty',
                    alpha=0.5, color='0.8', zorder=1
                )
                s2 = ax[l].fill_between(
                    x, drift_scan_se_min.isel(**selection),
                    drift_scan_se_max.isel(**selection),
                    label='Sample variance uncertainty',
                    alpha=0.5, color='0.4', zorder=2
                )
                ax[l].set_xlim(x[0], x[-1])
                if l > 0:
                    # ax[l].set_ylim(drift_scan_s.isel(**selection).min() * 1.25,
                    #                drift_scan_s.isel(**selection).max() * 1.25)
                    ax[l].axhline(0, c='black', ls='--', zorder=10)

            ax[0].yaxis.set_major_locator(MaxNLocator(2))
            ax[1].yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
            ax[2].yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            ax[0].set_ylim(0, 1)
            ax[1].set_ylim(-1.5, 2)
            ax[2].set_ylim(-1, 3.5)

            # Make upper x-axis with redshift coordinates
            axt_xticklabels = np.arange(3, 10) * 0.1
            axt_xticklocs = np.interp(axt_xticklabels, xi, fi)
            axt = ax[0].twiny()
            axt.set_xlim(ax[0].get_xlim())
            axt.set_xticks(axt_xticklocs)
            axt.set_xticklabels(axt_xticklabels)

            fig.text(0.02, 0.67, 'Variance', ha='left', va='center',
                     rotation='vertical')
            fig.text(0.02, 0.44, 'Skewness', ha='left', va='center',
                     rotation='vertical')
            fig.text(0.02, 0.21, 'Kurtotis', ha='left', va='center',
                     rotation='vertical')
            fig.text(0.5, 0.01, 'Observed Frequency [MHz]', va='bottom',
                     ha='center')
            fig.text(0.5, 0.86, 'Ionized Fraction', va='top', ha='center')

            fig.tight_layout(rect=[0.03, 0.02, 0.99, 0.85])

            lines = [l1[0], l2[0], l3[0], s2, s1]
            labels = [l.get_label() for l in lines]
            fig.legend(lines, labels, loc='upper center', ncol=2)

            out_dir = '{:s}/plots_v2/all_stats/{:s}/field{:d}'\
                .format(main_dir, telescope[j], signal_field)
            check_dir(out_dir)
            fig.savefig(
                '{:s}/all_stats_{:s}_field{:d}_bw{:.2f}MHz_{:s}.pdf'
                .format(out_dir, telescope[j], signal_field,
                        bandwidth[i], averaging[k])
            )
            # plt.show()
