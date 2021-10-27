import numpy as np


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
            trajectories = self._sample_arcs(params)

        elif mode == 'sines':
            trajectories = self._sample_sines(params)

        else:
            raise NotImplementedError

        return trajectories

    def _sample_arcs(self, params):

        raise NotImplementedError

    def _sample_sines(self, params):

        raise NotImplementedError

    def _collect(self, trajectories):

        if self.trajectories is None:
            self.trajectories = trajectories

        else:
            self.trajectories = np.concatenate(
                (self.trajectories, trajectories), axis=0)
