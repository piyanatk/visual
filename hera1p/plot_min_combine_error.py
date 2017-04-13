import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.close()

telescope = ['hera19', 'hera37', 'hera61', 'hera91',
             'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
# bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0]
stats = ['var', 'skew', 'kurt']
colors = ['#1b9e77', '#d95f02', '#7570b3']
markers = ['o', 's', '^']
markersize = [25, 50, 100, 200, 400]
data_dir = '/Users/piyanat/Google/research/hera1p/stats'
group = ['binning', 'windowing']


fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(group)):
    g = group[i]
    for j in range(len(bandwidth)):
        b = bandwidth[j]
        for k in range(len(telescope)):
            t = telescope[k]
            mc = pd.read_hdf('{:s}/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'.format(data_dir, t, t, b, g))
            hpx = pd.read_hdf('{:s}/healpix/{:s}/{:s}_hpx_interp_21cm_cube_l128_stats_df_bw{:.2f}MHz_{:s}.h5'.format(data_dir, t, t, b, g))

            # fid = np.random.randint(0, 200, 20)  # 20 fields
            fid = np.arange(200)  # 200 fields

            mean_stats = mc[fid].mean(axis=0)[stats]
            noise_err = mc[fid].mean(axis=0)[[s+'_err' for s in stats]]
            noise_err.columns = stats
            sample_err = mc[fid].std(axis=0)[stats]
            combine_err = np.sqrt(sample_err ** 2 + noise_err ** 2)
            x = mean_stats.index

            # for l in range(len(stats)):
            for l in [1]:
                s = stats[l]
                idx = x[np.where(combine_err[s] == combine_err[s].min())].values
                if g == 'binning':
                    ax.scatter(k+(j*0.1), idx[0], s=markersize[j], c=colors[l], marker=markers[l], alpha=0.5, edgecolors='face')
                else:
                    ax.scatter(k+(j*0.1), idx[0], s=markersize[j], c='none', marker=markers[l], alpha=0.5, edgecolors=colors[l])
ax.xaxis.set_ticks(np.arange(0, len(telescope)))
ax.xaxis.set_ticklabels(telescope, rotation=-45)
ax.set_xlim(-0.5, len(telescope)-0.5)
ax.set_ylim(138.915, 195.235)
ax.grid('on')
ax.set_title('Error Equivalent Point (Sample Error = Noise Error)')
ax.set_xlabel('HERA Configurations')
ax.set_ylabel('Frequency [MHz]')
plt.show()
