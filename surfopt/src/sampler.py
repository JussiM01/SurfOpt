import numpy as np

from src.utils import sample_arcs, sample_line, sample_sines, sample_sine_sums


class Sampler:

    def __init__(self, params):

        self.params = params
        self.paths = None

    def __call__(self):

        for mode in {'arcs', 'line', 'sines', 'sine_sums'}:
            if mode in self.params:
                sub_params = self.params[mode]
                sub_params['start'] = self.params['start']
                sub_params['end'] = self.params['end']
                paths = self._sample(mode, sub_params)
                self._collect(paths)

        self._fix_end_values()

        return self.paths

    def _sample(self, mode, params):

        if mode == 'arcs':
            paths = sample_arcs(params)

        elif mode == 'line':
            paths = sample_line(params)

        elif mode == 'sines':
            paths = sample_sines(params)

        elif mode == 'sine_sums':
            paths = sample_sine_sums(params)

        else:
            raise NotImplementedError

        return paths

    def _collect(self, paths):

        if self.paths is None:
            self.paths = paths.astype('float32')

        else:
            self.paths = np.concatenate(
                (self.paths, paths.astype('float32')), axis=0)

    def _fix_end_values(self):

        self.paths[:,0,:] = np.array(
            self.params['start'], dtype='float32')
        self.paths[:,-1,:] = np.array(
            self.params['end'], dtype='float32')



if __name__ == '__main__':

    import argparse

    import matplotlib.pyplot as plt

    from src.utils import unpack


    parser = argparse.ArgumentParser()

    parser.add_argument('-ua', '--use_arcs', action='store_true')
    parser.add_argument('-ul', '--use_line', action='store_true')
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
    parser.add_argument('-x0', '--start_x', type=float, default=-1.0)
    parser.add_argument('-y0', '--start_y', type=float, default=0.0)
    parser.add_argument('-x1', '--end_x', type=float, default=1.0)
    parser.add_argument('-y1', '--end_y', type=float, default=0.0)

    args = parser.parse_args()


    params = {
        'start': [args.start_x, args.start_y],
        'end': [args.end_x, args.end_y]
    }

    if args.use_arcs is True:

        params['arcs'] = {
            'num_angles': args.num_angles,
            'min_angle': args.min_angle,
            'max_angle': args.max_angle,
            'num_steps': args.num_steps
        }

    if args.use_line is True:

        params['line'] = {
            'num_steps': args.num_steps
        }

    if args.use_sines is True:

        params['sines'] = {
            'constants': unpack(args.constants, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_steps': args.num_steps
        }

    if args.use_sine_sums is True:

        params['sine_sums'] = {
            'up_ranges': unpack(args.up_ranges, 'float'),
            'multiples': unpack(args.multiples, 'int'),
            'num_samples': args.num_samples,
            'num_steps': args.num_steps
        }


    sampler = Sampler(params)
    paths = sampler()

    max_value = max(abs(np.min(paths)), np.max(paths))
    bound = max_value + 1

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.grid(True)

    for i in range(paths.shape[0]):
        x_values = paths[i, :, 0]
        y_values = paths[i, :, 1]
        ax.plot(x_values, y_values)

    plt.show()
