import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = '/Users/piyanat/Google/research/hera1p/stats'
mc = pd.read_hdf('{:s}/mc/hera331/hera331_mc_maps_stats_pn_bw0.08MHz_binning.h5'.format(data_dir))
hpx = pd.read_hdf('{:s}/healpix/hera331/hera331_hpx_interp_21cm_cube_l128_stats_df_bw0.08MHz_binning.h5'.format(data_dir))

# fid = np.random.randint(0, 200, 20)  # 20 fields
fid = np.arange(200)  # 200 fields

mean_stats = mc[fid].mean(axis=0)[['var', 'skew', 'kurt']]
noise_err = mc[fid].mean(axis=0)[['var_err', 'skew_err', 'kurt_err']]
noise_err.columns = ['var', 'skew', 'kurt']
sample_err = mc[fid].std(axis=0)[['var','skew','kurt']]

fig = plt.figure()
ax = fig.add_subplot(111)
(sample_err/noise_err).plot(ax=ax)
ax.axhline(1, color='k', ls='-.')
ax.grid('on')
ax.set_ylabel('Sample Error / Noise Error')
ax.set_xlabel('Frequency [MHz]')
plt.show()
