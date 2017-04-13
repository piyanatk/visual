import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from sim.settings import MWA_FREQ_EOR_ALL_80KHZ as F


cmap = colors.ListedColormap(
    ['#ffffd9',
     '#edf8b1',
     '#c7e9b4',
     '#7fcdbb',
     '#41b6c4',
     '#1d91c0',
     '#225ea8',
     '#253494',
     '#081d58']
    )

bounds = [0.07, 0.09, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

if __name__ == "__main__":
    plt.close()

    telescope = ['hera19', 'hera37', 'hera61', 'hera91',
                 'hera127', 'hera169', 'hera217', 'hera271', 'hera331']
    bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    stats = ['var', 'skew', 'kurt']
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    markers = ['o', 's', '^']
    markersize = [25, 50, 100, 200, 400]
    data_dir = '/Users/piyanat/Google/research/hera1p/stats'
    group = ['binning', 'windowing']

    # fid = np.random.randint(0, 200, 20)  # 20 fields
    fid = np.arange(200)  # 200 fields

    for i in range(len(stats)):
        s = stats[i]
        for j in range(len(group)):
            g = group[j]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            arr = np.zeros((9, 705))
            arr[:, :] = np.nan
            for k in range(len(telescope)):
                t = telescope[k]
                for l in range(len(bandwidth)):
                    b = bandwidth[-l-1]
                    print(s, b, g, t)
                    mc = pd.read_hdf('{:s}/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'.format(data_dir, t, t, b, g))
                    sample_err = mc[fid].std(axis=0)[s]
                    noise_err = mc[fid].mean(axis=0)[s+'_err']
                    x = noise_err.index.values
                    xrange = x[np.where(noise_err < sample_err)]
                    if xrange.size > 0:
                        idx = np.where((F > xrange.min()) & (F < xrange.max()))
                        arr[k, idx] = b
            print(s, g, np.nanmin(arr), np.nanmax(arr))
            im = ax.pcolorfast([F[0], F[-1]], [0, 9], arr, cmap=cmap, norm=norm)
            ax.set_title(s + ' ' + g)
            ax.yaxis.set_ticks(np.arange(0.5, 9, 1))
            ax.yaxis.set_ticklabels(telescope)
            # ax.set_xlabel(F)
            cb = fig.colorbar(im)
            cb.set_ticks(bandwidth)
            cb.set_ticklabels(bandwidth)
            cb.set_label('{:s} Size [MHz]'.format(g.capitalize()))
            fig.savefig('/Users/piyanat/Google/research/hera1p/plots/sample_error_chart/sample_error_chart_{:s}_{:s}.pdf'.format(s, g))
