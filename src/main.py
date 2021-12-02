import argparse

from optimizer import Optimizer


def main(params):

    optimizer = Optimizer(params['optimizer'])
    optimized_trajectory = optimizer(params['surface'], params['trajectories'])

    if params['print_best'] == True:
        print(optimized_trajectory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-no', '--num_opt_steps', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-pc', '--plot_changes', action='store_true')
    parser.add_argument('-pb', '--plot_best', action='store_true')
    parser.add_argument('-pr', '--plot_results', action='store_true')
    parser.add_argument('-s', '--save_plots', action='store_true')
    parser.add_argument('-o', '--optim_type', type=str, default='SGD')
    parser.add_argument('-b', '--bound', type=int, default=100) # CHANGE this ?
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
    parser.add_argument('-x0', '--start_x', type=float, default=-1.0)
    parser.add_argument('-y0', '--start_y', type=float, default=0.0)
    parser.add_argument('-x1', '--end_x', type=float, default=1.0)
    parser.add_argument('-y1', '--end_y', type=float, default=0.0)
    parser.add_argument('-pr', '--print_best', action='store_true')

    args = parser.parse_args()

    # surface_params = # (read from config-file)

    optimizer_params = {
        'num_opt_steps': args.num_opt_steps,
        'learning_rate': args.learning_rate,
        'plot_changes': args.plot_changes,
        'plot_best': args.plot_best,
        'plot_results': args.plot_results,
        'save_plots': args.save_plots,
        'optim_type': args.optim_type,
        'fig': = {'bound': args.bound} # CHANGE this and/or add more k,v pairs?
        }

    trajectory_params = {
        'use_arcs': args.use_arcs,
        'use_sines': args.use_sines,
        'use_sine_sums': args.use_sine_sums,
        'constants': args.constants,
        'multiples': args.multiples,
        'up_ranges': args.up_ranges,
        'num_samples': args.num_samples,
        'num_angles': args.num_angles,
        'min_angle': args.min_angle,
        'max_angle': args.max_angle,
        'num_steps': args.num_steps,
        'start_x': args.start_x,
        'start_y': args.start_y,
        'end_x': args.end_x,
        'end_y': args.end_y,
        'print_best': args.print_best
        }

    params = {
        'optimizer': optimizer_params,
        'surface': surface_params,
        'trajectories': trajectory_params,
        'print_best': args.print_best
        }

    main(params)
