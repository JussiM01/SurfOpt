import argparse
import numpy as np
import os
import time

from src.optimizer import Optimizer
from src.randomsurface import create
from src.utils import load, unpack, write
from src.viewsurface import view


def main(params):

    if params['view_surface']:
        view(params)

    elif params['create_surface']:
        create(params)

    else:
        optimizer = Optimizer(params['optimizer'])
        optimized_path = optimizer(
            params['surface'], params['paths'])

        if params['print_best'] == True:
            print('\nOPTIMIZED PATH:\n\n{}\n'.format(
                optimized_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--conf_file', type=str,
        default='gauss2hills')
    parser.add_argument('-v', '--view_surface', action='store_true')
    parser.add_argument('-cs', '--create_surface', action='store_true')
    parser.add_argument('-cm', '--cmap', type=str, default='viridis')
    parser.add_argument('-no', '--num_opt_steps', type=int, default=1000)
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-3)
    parser.add_argument('-re', '--regular_const', type=float, default=1e-2)
    parser.add_argument('-pa', '--plot_all', action='store_true')
    parser.add_argument('-pc', '--plot_changes', action='store_true')
    parser.add_argument('-pb', '--plot_best', type=bool, default=True)
    parser.add_argument('-pr', '--plot_results', action='store_true')
    parser.add_argument('-s', '--save_plots', action='store_true')
    parser.add_argument('-o', '--optim_type', type=str, default='SGD')
    parser.add_argument('-bx', '--bound_x', type=float, default=5.0) # CHANGE this ?
    parser.add_argument('-by', '--bound_y', type=float, default=2.5) # CHANGE this ?
    parser.add_argument('-ul', '--use_line', type=bool, default=True)
    parser.add_argument('-ua', '--use_arcs', action='store_true')
    parser.add_argument('-us', '--use_sines', action='store_true')
    parser.add_argument('-uss', '--use_sine_sums', action='store_true')
    parser.add_argument('-c', '--constants', type=str, default='1.0,1.0,1.0')
    parser.add_argument('-m', '--multiples', type=str, default='1,-1')
    parser.add_argument('-u', '--up_ranges', type=str, default='1.0,1.0,1.0')
    parser.add_argument('-nsa', '--num_samples', type=int, default=10)
    parser.add_argument('-na', '--num_angles', type=int, default=1)
    parser.add_argument('-mi', '--min_angle', type=int, default=np.pi/18)
    parser.add_argument('-ma', '--max_angle', type=int, default=np.pi/4)
    parser.add_argument('-ns', '--num_steps', type=int, default=50)
    parser.add_argument('-x0', '--start_x', type=float, default=-2.0)
    parser.add_argument('-y0', '--start_y', type=float, default=0.0)
    parser.add_argument('-x1', '--end_x', type=float, default=2.0)
    parser.add_argument('-y1', '--end_y', type=float, default=0.0)
    parser.add_argument('-prb', '--print_best', type=bool, default=True)
    parser.add_argument('-xmi', '--x_min', type=float, default=-5.0)
    parser.add_argument('-xma', '--x_max', type=float, default=5.0)
    parser.add_argument('-ymi', '--y_min', type=float, default=-2.5)
    parser.add_argument('-yma', '--y_max', type=float, default=2.5)
    parser.add_argument('-xs', '--x_size', type=int, default=50)
    parser.add_argument('-ys', '--y_size', type=int, default=50)
    parser.add_argument('-r', '--random_seed', type=int)
    parser.add_argument('-ng', '--num_gauss', type=int, default=10)
    parser.add_argument('-d0', '--diag_min', type=float, default=5.0)
    parser.add_argument('-d1', '--diag_max', type=float, default=10.0)
    parser.add_argument('-o0', '--offd_min', type=float, default=-5.0)
    parser.add_argument('-o1', '--offd_max', type=float, default=5.0)
    parser.add_argument('-sc', '--scale', type=float, default=1e3)
    parser.add_argument('-sp', '--save_params', action='store_true')
    parser.add_argument('-sn', '--saving_name', type=str, default=None)
    parser.add_argument('-pf', '--params_file', type=str, default=None)

    args = parser.parse_args()

    # set random seed
    np.random.seed(args.random_seed)

    surface_params = load(args.conf_file + '.json', 'config_files')

    if args.saving_name is None:
        saving_name = time.strftime("%Y_%m_%d_%Z_%H_%M_%S")

    else:
        saving_name = args.saving_name

    optimizer_params = {
        'num_opt_steps': args.num_opt_steps,
        'learning_rate': args.learning_rate,
        'regular_const': args.regular_const,
        'plot_changes': args.plot_changes,
        'plot_best': args.plot_best,
        'plot_results': args.plot_results,
        'plot_all': args.plot_all,
        'save_plots': args.save_plots,
        'optim_type': args.optim_type,
        'saving_name': saving_name,
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

    path_params = {
        'start': [args.start_x, args.start_y],
        'end': [args.end_x, args.end_y]
    }

    if args.use_arcs is True:

        path_params['arcs'] = {
            'num_angles': args.num_angles,
            'min_angle': args.min_angle,
            'max_angle': args.max_angle,
            'num_steps': args.num_steps
        }

    if args.use_line is True:

        path_params['line'] = {
            'num_steps': args.num_steps
        }

    if args.use_sines is True:

        path_params['sines'] = {
            'constants': unpack(args.constants, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_steps': args.num_steps
        }

    if args.use_sine_sums is True:

        path_params['sine_sums'] = {
            'up_ranges': unpack(args.up_ranges, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_samples': args.num_samples,
            'num_steps': args.num_steps
        }

    if args.view_surface:
        params = {
            'view_surface': True,
            'create_surface': False,
            'conf_file': args.conf_file + '.json',
            'grid': {
                'x_min': args.x_min,
                'x_max': args.x_max,
                'y_min': args.y_min,
                'y_max': args.y_max,
                'x_size': args.x_size,
                'y_size': args.y_size
                },
            'cmap': args.cmap
            }

    elif args.create_surface:
        params = {
            'view_surface': False,
            'create_surface': True,
            'random_seed': args.random_seed,
            'num_gauss': args.num_gauss,
            'x_min': args.x_min,
            'x_max': args.x_max,
            'y_min': args.y_min,
            'y_max': args.y_max,
            'diag_min': args.diag_min,
            'diag_max': args.diag_max,
            'offd_min': args.offd_min,
            'offd_max': args.offd_max,
            'scale': args.scale
            }

    else:
        params = {
            'view_surface': False,
            'create_surface': False,
            'optimizer': optimizer_params,
            'surface': surface_params,
            'paths': path_params,
            'print_best': args.print_best
            }

    if args.save_params:
        write(params, saving_name + '.json', 'saved_params')

    elif args.params_file is not None:
        params = load(args.params_file + '.json', 'saved_params')

    main(params)
