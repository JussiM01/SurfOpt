import numpy as np

from utils import sample_arcs, sample_sines


class Sampler:

    def __init__(self, params):

        self.params = params
        self.trajectories = None

    def __call__(self):

        for mode in self.params:
            sub_params = self.params[mode]
            trajectories = self._sample(mode, sub_params)
            self._collect(trajectories)

        return self.trajectories

    def _sample(self, mode, params):

        if mode == 'arcs':
            trajectories = sample_arcs(params)

        elif mode == 'sines':
            trajectories = sample_sines(params)

        else:
            raise NotImplementedError

        return trajectories

    def _collect(self, trajectories):

        if self.trajectories is None:
            self.trajectories = trajectories

        else:
            self.trajectories = np.concatenate(
                (self.trajectories, trajectories), axis=0)


if __name__ == '__main__':

    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument('-ua', '--use_arcs', type=bool, default=True)
    parser.add_argument('-na', '--num_angles', type=int, default=9)
    parser.add_argument('-mi', '--min_angle', type=int, default=np.pi/18)
    parser.add_argument('-ma', '--max_angle', type=int, default=np.pi/18)
    parser.add_argument('-ns', '--num_steps', type=int, default=50)

    args = parser.parse_args()

    if args.use_arcs is True:

        params = {
            'num_angles': args.num_angles,
            'min_angle': args.min_angle,
            'max_angle': args.max_angle,
            'num_steps': args.num_steps
        }

    elif:
         raise NotImplementedError

    sampler = Sampler(params)
    trajectories = sampler()

    # Plot the trajectories here.
