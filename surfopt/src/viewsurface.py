import matplotlib.pyplot as plt

from src.surfacemap import SurfaceMap
from src.utils import create_grid, load


def view(params):

    data = load(params['conf_file'], 'config_files')
    surfacemap = SurfaceMap(data)
    X, Y, Z = create_grid(params['grid'], surfacemap)
    fig = plt.figure(figsize=(7, 9))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=params['cmap'],
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=30, location='left')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.contourf(X, Y, Z, cmap=params['cmap'])
    fig.suptitle('File: ' + params['conf_file'])

    plt.show()
