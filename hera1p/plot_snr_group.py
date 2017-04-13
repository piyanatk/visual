import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

data_dir = '/Users/piyanat/Google/research/hera1p/stats/mc'
telescope = ['hera19', 'hera37', 'hera61', 'hera91', 'hera127',
             'hera169', 'hera217', 'hera271', 'hera331']
bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
stat = ['var', 'skew', 'kurt']
group = ['binning', 'windowing']

nbadwidth = len(bandwidth)
nstat = len(stat)
nfield = 20
if nfield < 200:
    fid = np.random.randint(0, 200, nfield)
else:
    fid = np.arange(200)

ls = ['-', '--', '-.', '-', '-', '-', '-', '-', '-']
m = ['None', 'None', 'None', 'x', '*', 'o', 's', '^', 'D']
color = [
    '#67000d',
    '#a50026',
    '#d73027',
    '#f46d43',
    '#fdae61',
    '#abd9e9',
    '#74add1',
    '#4575b4',
    '#313695',
    '#08306b'
]
ylabel1 = ['$S_2$', '$S_3$', '$S_4$']
# ylabel2 = [r'$\frac{|S_2|}{\sigma_{n,S_2}}$',
#            r'$\frac{|S_3|}{\sigma_{n,S_3}}$',
#            r'$\frac{|S_4|}{\sigma_{n,S_4}}$']
ylabel2 = ['$SNR_{S_2}$', '$SNR_{S_3}$', '$SNR_{S_4}$']
ylim1 = [(0, 1), (-1, 1.5), (-1, 2.5)]
ylim2 = [(0, 40), (0, 16), (0, 14)]
tlabels = ['HERA19', 'HERA37', 'HERA61', 'HERA91', 'HERA128',
           'HERA169', 'HERA240 Core', 'HERA271', 'HERA350 Core']

arr = np.zeros((9, 3, 705))

for i in range(nbadwidth):
    bw = bandwidth[i]
    df = pd.read_hdf(
        '{:s}/hera331/hera331_mc_maps_stats_df_bw{:.2f}MHz_windowing.h5'
        .format(data_dir, bw)
    )
    for j in range(nstat):
        s = stat[j]
        arr[i, j, :] = df[s+'_mc_err'].values