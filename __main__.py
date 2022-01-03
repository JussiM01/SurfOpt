import argparse
import numpy as np
import os

from src.optimizer import Optimizer
from src.utils import load_config, unpack


def main(params):

    optimizer = Optimizer(params['optimizer'])
    optimized_trajectory = optimizer(params['surface'], params['trajectories'])

    if params['print_best'] == True:
        print(optimized_trajectory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--conf_file', type=str,
        default='gauss2hills.json')
    parser.add_argument('-no', '--num_opt_steps', type=int, default=1000)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-pc', '--plot_changes', action='store_true')
    parser.add_argument('-pb', '--plot_best', action='store_true')
    parser.add_argument('-pr', '--plot_results', action='store_true')
    parser.add_argument('-s', '--save_plots', action='store_true')
    parser.add_argument('-o', '--optim_type', type=str, default='SGD')
    parser.add_argument('-bx', '--bound_x', type=float, default=5.0) # CHANGE this ?
    parser.add_argument('-by', '--bound_y', type=float, default=2.5) # CHANGE this ?
    parser.add_argument('-ua', '--use_arcs', action='store_true')
    parser.add_argument('-us', '--use_sines', action='store_true')
    parser.add_argument('-uss', '--use_sine_sums', action='store_true')
    parser.add_argument('-c', '--constants', type=str, default='1.0,1.0,1.0')
    parser.add_argument('-m', '--multiples', type=str, default='0,1,-1')
    parser.add_argument('-u', '--up_ranges', type=str, default='1.0,1.0,1.0')
    parser.add_argument('-nsa', '--num_samples', type=int, default=10)
    parser.add_argument('-na', '--num_angles', type=int, default=4)
    parser.add_argument('-mi', '--min_angle', type=int, default=np.pi/18)
    parser.add_argument('-ma', '--max_angle', type=int, default=np.pi/4)
    parser.add_argument('-ns', '--num_steps', type=int, default=50)
    parser.add_argument('-x0', '--start_x', type=float, default=-2.0)
    parser.add_argument('-y0', '--start_y', type=float, default=0.0)
    parser.add_argument('-x1', '--end_x', type=float, default=2.0)
    parser.add_argument('-y1', '--end_y', type=float, default=0.0)
    parser.add_argument('-prb', '--print_best', action='store_true')
    parser.add_argument('-xmi', '--x_min', type=float, default=-5.0)
    parser.add_argument('-xma', '--x_max', type=float, default=5.0)
    parser.add_argument('-ymi', '--y_min', type=float, default=-2.5)
    parser.add_argument('-yma', '--y_max', type=float, default=2.5)
    parser.add_argument('-xs', '--x_size', type=int, default=50)
    parser.add_argument('-ys', '--y_size', type=int, default=50)



    args = parser.parse_args()

    surface_params = load_config(args.conf_file)

    optimizer_params = {
        'num_opt_steps': args.num_opt_steps,
        'learning_rate': args.learning_rate,
        'plot_changes': args.plot_changes,
        'plot_best': args.plot_best,
        'plot_results': args.plot_results,
        'save_plots': args.save_plots,
        'optim_type': args.optim_type,
        'fig': { # CHANGE these and/or add more k,v pairs ?
            'grid': {
                'x_min': args.x_min,
                'x_max': args.x_max,
                'y_min': args.y_min,
                'y_max': args.y_max,
                'x_size': args.x_size,
                'y_size': args.y_size
                },
            'plot': {
                'bound': {
                    'x': args.bound_x,
                    'y': args.bound_y
                    }
                }
            }
        }

    trajectory_params = {
        'start': [args.start_x, args.start_y],
        'end': [args.end_x, args.end_y]
    }

    if args.use_arcs is True:

        trajectory_params['arcs'] = {
            'num_angles': args.num_angles,
            'min_angle': args.min_angle,
            'max_angle': args.max_angle,
            'num_steps': args.num_steps
        }

    if args.use_sines is True:

        trajectory_params['sines'] = {
            'constants': unpack(args.constants, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_steps': args.num_steps
        }

    if args.use_sine_sums is True:

        trajectory_params['sine_sums'] = {
            'up_ranges': unpack(args.up_ranges, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_samples': args.num_samples,
            'num_steps': args.num_steps
        }

    params = {
        'optimizer': optimizer_params,
        'surface': surface_params,
        'trajectories': trajectory_params,
        'print_best': args.print_best
        }

    main(params)
