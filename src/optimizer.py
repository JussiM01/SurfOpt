import torch

from sampler import Sampler
from surfacemap import SurfaceMap


class Optimizer:

    def __init__(self, params):

        self.num_steps = params['num_steps']
        self.learning_rate = params['learning_rate']
        self.plot_changes = params['plot_changes']
        self.plot_results = params['plot_results']
        self.save_plots = params['save_plots']
        self.optim_type = params['optim_type']

    def __call__(surface_params, trajectory_params):

        self._initialize(surface_params, trajectory_params)

        for i in range(self.num_steps):
            self._optim_step()

        if plot_changes:
            self._create_changes_plot()

        if self.plot_results:
            self._create_results_plot()

        # Return here the trajectory which has the lowest lenght
        # after the optimization.

    def _initialize(self, surface_params, trajectory_params):

        sampler = Sampler(trajectory_params)
        trajectories = sampler()
        inside_trajs = torch.from_numpy(trajectories[:,1:-1,:])
        self.surfacemap = SurfaceMap(surface_params)
        self.optimizer = self._set_optimizer(inside_trajs)
        self.start_h = self.surfacemap(torch.from_numpy(trajectories[:,0:1,:))
        self.end_h = self.surfacemap(torch.from_numpy(trajectories[:,-1:,:]))

    def _set_optimizer(self, inside_trajs):

        if self.optim_type == 'Adam':
            return torch.optim.Adam([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'AdamW':
            return torch.optim.AdamW([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'SGD':
            return torch.optim.SGD([inside_trajs], lr=self.learning_rate)

    def _optim_step(self):

        raise NotImplementedError

    def _create_changes_plot(self):

        raise NotImplementedError

    def _create_results_plot(self):

        raise NotImplementedError
