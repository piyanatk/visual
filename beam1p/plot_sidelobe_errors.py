from glob import glob
import numpy as np
# import matplotlib
# matplotlib.use('agg')
# from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.pyplot import Line2D
# from matplotlib.ticker import MaxNLocator
import pandas as pd


stats_dir = '/Users/piyanat/research/pdf_paper/stats/'
fhd = pd.read_hdf(stats_dir + 'mwa128_fhd_stats_df_bw0.08MHz.h5')
gauss = pd.read_hdf(stats_dir + 'mwa128_gauss_stats_df_bw0.08MHz.h5')
# fhd['skew'].plot(style='b')
# gauss['skew'].plot(style='g')
# a = fhd['skew'] / fhd['var']
# b = gauss['skew'] / fhd['var']
# (fhd['skew'] / fhd['var']).plot(style='b:')
# (gauss['skew'] / fhd['var']).plot(style='g:')
# plt.figure()
# plt.plot(np.abs(a-b))
x = fhd.index.values

# y0 = (gauss['var'] / fhd['var'].values) - (fhd['var'] / fhd['var'].values)
# y1 = (gauss['skew'] / np.sqrt(fhd['var'].values)) - (fhd['var'] / np.sqrt(fhd['var'].values))
# y2 = (gauss['kurt'] / fhd['var'].values) - (fhd['var'] / fhd['var'].values)
y0 = np.abs(fhd['var'] - gauss['var']) / np.abs(gauss['var'])
y0 /= y0.std()
y1 = np.abs(fhd['skew'] - gauss['skew']) / np.abs(gauss['skew'])
y1 /= y1.std()
y2 = np.abs(fhd['kurt'] - gauss['kurt']) / np.abs(gauss['kurt'])
y2 /= y2.std()
plt.plot(x, y0, label='variance')
plt.plot(x, y1, label='skewness')
plt.plot(x, y2, label='kurtosis')
plt.legend()
# print(errors[['var', 'skew', 'kurt']].mean())
# print(errors[['var', 'skew', 'kurt']].std())
# y0 = errors['var'].values
# y1 = errors['skew'].values
# y2 = errors['kurt'].values
# fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
# ax[0].plot(x, y0, 'k:')
# ax[1].plot(x, y1, 'k--')
# ax[2].plot(x, y2, 'k-')
# errors.plot(y=['var', 'skew', 'kurt'], legend=True, subplots=True,
#             sharex=True, sharey=True, style=['k:', 'k--', 'k-'])
# ax[0].set_xlim(x[0], x[-1])
# plt.plot(x, y0, 'b-', label='Variance')
# plt.plot(x, y1, 'g--', label='Skewness')
# plt.plot(x, y2, 'r:', label='Kurtosis')
# plt.xlim(x[0], x[-1])
plt.show()