import xarray as xr
import matplotlib.pyplot as plt


ds = xr.open_dataset('/Users/piyanat/Google/data/hera1p/model_pdf.nc')
idx = [0, 121, 241, 361, 473, 564, 687]  # 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95

linestyles = [(25, (30, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
              (20, (25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
              (15, (20, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
              (10, (15, 1, 1, 1, 1, 1, 1, 1)),
              (5, (10, 1, 1, 1, 1, 1)), (0, (5, 1, 1, 1)), '-']
labels = ['$x_i={:.2f},\\ z={:.2},\\ \\nu_{{obs}}={:.3f}\\,MHz$'
          .format(ds.xi.values[i], ds.z.values[i], ds.f.values[i]) for i in idx]
plt.close()
fig, ax = plt.subplots(figsize=(5, 3.5))
for i in range(len(idx)):
    ax.plot(ds['bincen'].isel(f=idx[i]), ds['pdf'].isel(f=idx[i]),
            label=labels[i], ls=linestyles[i], alpha=1)
ax.set_xlim(-1, 21)
ax.set_ylim(0, 0.7)
ax.set_xlabel('Brightness Temperature [mK]')
ax.set_ylabel('$T_b\ dP/dT_b$')
plt.legend(handlelength=3, loc=(0.06, 0.43))
plt.tight_layout()
fig.savefig('/Users/piyanat/Google/data/hera1p/plots_v2/model_pdf.pdf')
