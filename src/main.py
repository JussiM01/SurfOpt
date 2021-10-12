import argparse

from optimizer import Optimizer
from utils import create_countours, create_plot


def main(params):

    # Make contours grid of the surfacemap.

    optimizer = Optimizer(params['optimizer'])
    results = optimizer(params['surface'], params['trajectories'])

    # if params['visualize_opt'] == True:

        # Plot the initial trajectories.

        # For each trajectory index and for each optimization step index
        # plot the corresponding trajectory.

    # Plot the best trajectory from the last optimization step.


# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser()
    #
    # parser.add_argument(...)
    # parser.add_argument(...) ...

    # surface_params = # (read from config-file)

    # params = {
    #     'optimizer': ...,
    #     'surface': surface_params,
    #     'trajectories': ...
    #     }

    # main(params)
