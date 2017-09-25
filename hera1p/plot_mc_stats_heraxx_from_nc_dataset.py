import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


main_dir = '/Users/piyanat/Google/data/hera1p'
data_dir = '{:s}/stats'.format(main_dir)
telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
             'hera169', 'hera217', 'hera271', 'hera331']
# bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 8.0]
bandwidth = 1.0
# stat = ['var', 'skew', 'kurt']
stat = 'kurt'

data = xr.open_dataset(
    data_dir + '/hera1p_all_stats_bw{:.2f}MHz.nc'.format(bandwidth)
)

stats = data['single_field_stats']
mean_stats = stats.mean(dim='field')
errors = data['single_field_sample_errors']
x = stats.frequency.values

s_dict = dict(var=0, skew=1, kurt=2)
s = s_dict[stat]

nrows = len(telescope)
ncols = 5
npages = int(200 / ncols)

for i in range(npages):
    fields = np.arange(ncols * i, ncols * i + ncols)  # field number in page i
    plt.close()
    fig, ax = plt.subplots(nrows=len(telescope), ncols=ncols, sharex='all',
                           sharey='all', figsize=(13, 8))
    for j in range(nrows):  # Rows, telescopes
        for k in range(ncols):  # Columns, fields
            f = ncols * i + k
            cuts1 = dict(telescope=j, field=f, stat=s, averaging=1)
            cuts2 = dict(telescope=j, stat=s, averaging=1)
            y = stats.isel(**cuts1).values
            ymean = mean_stats.isel(**cuts2).values
            yerr_min = ymean - errors.isel(**cuts2).values
            yerr_max = ymean + errors.isel(**cuts2).values

            ax[j, k].axhline(0, color='black', linestyle='--', linewidth=0.8,
                             zorder=1)
            l1 = ax[j, k].plot(x, y, linestyle='-', alpha=0.8, zorder=1)
            l2 = ax[j, k].plot(x, ymean, color='black', linestyle=':',
                               alpha=0.8, zorder=2)
            e1 = ax[j, k].fill_between(x, yerr_min, yerr_max, edgecolors='None',
                                       facecolors='0.5', alpha=0.5, zorder=2)
            if j == 0:
                ax[j, k].set_title('Field {:d}'.format(f), fontsize='small')
            if k == ncols-1:
                ax[j, k].yaxis.set_label_position('right')
                ax[j, k].set_ylabel('{:s}'.format(telescope[j].upper()),
                                    fontsize='small')
    ax[0, 0].set_ylim(-1, 2.5)
    ax[0, 0].set_xlim(x[0], x[-1])
    lines = [l1[0], l2[0], e1]
    legend_labels = ['Single Field Kurtosis', 'All Fields Mean Kurtosis',
                     'Single Field Sample Variance']
    fig.text(0.02, 0.5, 'Kurtosis', rotation='vertical', ha='left', va='center')
    fig.text(0.25, 0.02, 'Observed Frequency [MHz]', ha='center', va='bottom')
    fig.legend(handles=lines, labels=legend_labels, loc='lower right', ncol=3)
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.08, top=0.95,
                        hspace=0, wspace=0)
    outdir = main_dir + '/plots_v2/mc_stats_heraxx/bw{:.2f}MHz/{:s}'\
        .format(bandwidth, stat)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fig.savefig(
        '{:s}/mc_stats_heraxx_{:s}_bw{:.2f}_windowing_field{:03d}-{:03d}.pdf'
        .format(outdir, stat, bandwidth, fields[0], fields[-1])
    )
    # plt.show()
