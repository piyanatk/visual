from numpy import ma
from matplotlib import cbook
from matplotlib import colors
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint


norm = MidPointNorm(midpoint=0)

if __name__ == "__main__":
    data_dir = '/Users/piyanat/Google/research/hera1p/stats'

    telescope = ['hera19', 'hera37', 'hera61', 'hera91',
                 'hera127', 'hera169', 'hera217', 'hera271', 'hera331']

    # bandwidth = [0.08, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    bandwidth = [0.08]

    # stats = ['var', 'skew', 'kurt']
    stats = ['kurt']

    # group = ['binning', 'windowing']
    group = ['binning']

    nfield = 200
    if nfield < 200:
        fid = np.random.randint(0, 200, 20)  # 20 fields
    else:
        fid = np.arange(200)  # 200 fields

    for i in range(len(stats)):
        s = stats[i]
        for j in range(len(group)):
            g = group[j]
            for k in range(len(telescope)):
                t = telescope[k]
                for l in range(len(bandwidth)):
                    b = bandwidth[l]
                    plt.close()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    print(s, b, g, t)
                    mc = pd.read_hdf('{:s}/mc/{:s}/{:s}_mc_maps_stats_pn_bw{:.2f}MHz_{:s}.h5'.format(data_dir, t, t, b, g))
                    sample_err = mc[fid].std(axis=0)[s]
                    mean_signal = mc[fid].mean(axis=0)[s]
                    x = sample_err.index.values
                    arr = np.zeros((200, x.size))
                    for m in range(200):
                        arr[m, :] = (mc.iloc[m][s] - mean_signal)/sample_err
                    im = ax.pcolorfast([x[0], x[-1]], [0, 200], arr, cmap='coolwarm', norm=norm)
                    # ax.set_title(s + ' ' + g)
                    # ax.yaxis.set_ticks(np.arange(0.5, 9, 1))
                    # ax.yaxis.set_ticklabels(telescope)
                    # ax.set_xlabel(F)
                    cb = fig.colorbar(im)
                    # cb.set_ticks(bandwidth)
                    # cb.set_ticklabels(bandwidth)
                    # cb.set_label('{:s} Size [MHz]'.format(g.capitalize()))
                    fig.savefig('/Users/piyanat/Google/research/hera1p/plots/'
                                'residual_chart/'
                                'residual_chart_{:s}_{:s}_bw{:.2f}MHz_{:s}.pdf'
                                .format(t, s, b, g))
