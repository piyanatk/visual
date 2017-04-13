import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from astropy.modeling import models


freqs, zi, xi = np.genfromtxt(
    '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True
)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--recalculate', action='store_true',
    #                     help='recalculte PSF')
    # args = parser.parse_args()
    # if args.recalculate:
    #     uvall = fits.getdata(
    #         '/data3/piyanat/runs/shared/'
    #         'vis_interp_delta_21cm_l128_0.000h_195.075MHz_UV_weights_XX.fits'
    #     )
    #     uvlt50 = fits.getdata(
    #         '/data3/piyanat/runs/fhd_uvlt50/output_data/'
    #         'vis_interp_delta_21cm_l128_0.000h_195.235MHz_UV_weights_XX.fits'
    #     )
    #     psf1 = fftshift(fftn(fftshift(uvall))).real
    #     psf2 = fftshift(fftn(fftshift(uvlt50))).real
    #     degpix = 1 / (7480 * 0.477465) * 180 / np.pi
    #     x = np.arange(-3740, 3740) * degpix
    #     y1 = psf1[3740, :] / psf1.max()
    #     y2 = psf2[3740, :] / psf2.max()
    # else:
    x, y1, y2 = np.genfromtxt('/Users/piyanat/Google/research/pdf_paper/psf/psf.csv',
                              delimiter=',', unpack=True)
    amp, mean, std = np.genfromtxt(
        '/Users/piyanat/Google/research/pdf_paper/psf/psf_fit_params_195.235MHz.csv',
        delimiter=','
    )

    gauss = models.Gaussian1D(amp, mean, std)
    x3 = np.linspace(-15, 15, 1000)
    y3 = gauss(x3)
    x4 = np.hstack((np.arange(-15, -1.5, 2), np.arange(-1.4, 1.5, 0.2), np.arange(1.4, 15, 2)))
    y4 = gauss(x4)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y1, '-', color='0.5')
    ax.plot(x, y2, 'k:')
    ax.plot(x3, y3, '-', color='k', linewidth=0.5)
    ax.plot(x4, y4, linestyle='none', marker='s', markerfacecolor='none')
    ax.set_yscale('symlog', linthreshy=0.1, linscaley=1.5)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.04, 1.5)
    ax.yaxis.set_major_locator(FixedLocator([-0.05, 0, 0.05, 0.1, 0.5, 1.0]))
    ax.yaxis.set_minor_locator(FixedLocator(
        [-0.06, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.06,
         0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    )
    ax.yaxis.set_ticklabels(['-0.05', '0', '0.05', '0.1', '0.5', '1'])
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.grid('on')
    handlers = [Line2D([], [], linestyle='-', color='0.5'),
                Line2D([], [], linestyle=':', color='black'),
                Line2D([], [], linestyle='-', color='k', linewidth=0.5,
                       marker='s', markerfacecolor='none')]
    labels = ['MWA Phase I\n(all baselines)', 'MWA Phase I Core\n(baseline < 100 m)', 'Gaussian Fitted\nMWA Phase I Core']
    ax.legend(handlers, labels, loc='best')
    ax.set_xlabel('Radial Distance [Degree]')
    ax.set_ylabel('Relative Intensity')
    plt.tight_layout()
    fig.savefig('psf.pdf', dpi=200)
