import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
import pandas as pd


freqs, zi, xi = np.genfromtxt(
    '/Users/piyanat/research/pdf_paper/interp_delta_21cm_f_z_xi.csv',
    delimiter=',', unpack=True
)

if __name__ == '__main__':
    # Data parameters
    stats_dir = '/Users/piyanat/research/pdf_paper/stats/'
    df = pd.read_hdf(stats_dir + 'model_interp_l128_stats_df.h5')

    # Init figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axt_sub = ax.twinx()
    axt = axt_sub.twiny()

    # Plot
    df.plot(x='freqs', y=['var', 'skew', 'kurt'], style=['k--', 'k-.', 'k:'],
            ax=ax, legend=False, linewidth=2)
    ax.set_xlim(freqs[0], freqs[-1])
    axt.plot(zi, xi, ls='-', c='0.5', linewidth=2)
    axt.set_xlim(zi[0], zi[-1])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.minorticks_on()

    # Add twin axes at the top
    axt.spines['left'].set_visible(False)
    axt.spines['bottom'].set_visible(False)
    axt.xaxis.set_ticks_position('top')
    axt.xaxis.set_label_position('top')
    axt.yaxis.set_ticks_position('right')
    axt.yaxis.set_label_position('right')
    # axt.spines['top'].set_color('0.5')
    axt.spines['right'].set_color('0.5')
    # axt.tick_params(axis='x', which='both', colors='0.5')
    axt_sub.tick_params(axis='y', which='both', colors='0.5')
    axt_sub.yaxis.label.set_color('0.5')
    # axt.xaxis.label.set_color('0.5')
    axt.minorticks_on()

    # Axes labels
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Statistical Values')
    axt.set_xlabel('Redshift')
    axt_sub.set_ylabel('Ionized Fraction')

    # Legend
    handlers = [
        Line2D([], [], linestyle='-', color='0.5', linewidth=2),
        Line2D([], [], linestyle='--', color='black', linewidth=2),
        Line2D([], [], linestyle='-.', color='black', linewidth=2),
        Line2D([], [], linestyle=':', color='black', linewidth=2),
    ]
    labels = ['Ionized Fraction', 'Variance', 'Skewness', 'Kurtosis']
    ax.legend(handles=handlers, labels=labels, loc=(0.25, 0.5),
              fontsize='medium')

    # Tidy up
    plt.tight_layout()
    fig.canvas.draw()
    # plt.grid('on')
    fig.savefig('model_stats_l128.pdf', dpi=200)
