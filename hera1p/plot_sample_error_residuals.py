import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import pandas as pd
from astropy.modeling import fitting, powerlaws


def calculate_residual_error(stat_panel):
    pn = pd.read_hdf(stat_panel)
    indv_field_stats = pn[:, :, ['var', 'skew', 'kurt']]
    sky_avg_stats = indv_field_stats.mean(axis=0).values
    field_avg_stats = np.empty((200, 705, 3))
    for i in range(200):
        field_avg_stats[i, :, :] = indv_field_stats.iloc[
            np.random.randint(0, 200, i+1)
        ].mean(axis=0)
    stats_subtracted = field_avg_stats - sky_avg_stats
    return stats_subtracted.std(axis=1)


def fit_res(x, res):
    fitter = fitting.LevMarLSQFitter()
    p1_init = powerlaws.PowerLaw1D(res[:, 0].max(), 1, 0.5)
    p2_init = powerlaws.PowerLaw1D(res[:, 1].max(), 1, 0.5)
    p3_init = powerlaws.PowerLaw1D(res[:, 2].max(), 1, 0.5)
    p1_init.alpha.fixed = True
    p2_init.alpha.fixed = True
    p3_init.alpha.fixed = True
    p1_fit = fitter(p1_init, x, res[:, 0])
    p2_fit = fitter(p2_init, x, res[:, 1])
    p3_fit = fitter(p3_init, x, res[:, 2])
    return p1_fit, p2_fit, p3_fit


def plot_residual(res, out, title=None):
    x = np.arange(1, 201)
    p1, p2, p3 = fit_res(x, res)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, res[:, 0], 'g-', alpha=0.5, label='var')
    ax.plot(x, res[:, 1], 'b-', alpha=0.5, label='skew')
    ax.plot(x, res[:, 2], 'k-', alpha=0.5, label='kurt')
    ax.plot(x, p1(x), 'g--',
            label='var: $1/\sqrt{N}$ fit')
    ax.plot(x, p2(x), 'b--',
            label='skew: $1/\sqrt{N}$ fit')
    ax.plot(x, p3(x), 'k--',
            label='kurt: $1/\sqrt{N}$ fit')
    ax.set_xlabel('Number of Random Field (N)')
    ax.set_ylabel('$\sigma_{\mathrm{residual}}$', size='large')
    ax.set_title(title)
    ax.legend()
    fig.savefig(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('stat_panel', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()
    plot_residual(
        calculate_residual_error(args.stat_panel),
        args.outfile,
        title=args.outfile.split('/')[-1][:-3].replace('mc_maps_stats_pn', '')
    )
