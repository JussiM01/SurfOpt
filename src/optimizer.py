import torch

from sampler import Sampler
from surfacemap import SurfaceMap
from torch.nn import MSELoss


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inside_trajs = torch.from_numpy(
            trajectories[:,1:-1,:]).to(device).requires_grad_(True)
        self.surfacemap = SurfaceMap(surface_params)
        self.optimizer = self._set_optimizer(self.inside_trajs)
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

        loss = 0
        inside_hs = self.surfacemap(self.inside_trajs)
        loss += MSELoss(self.start_h - inside_hs[:,0:1,:])
        loss += MSELoss(inside_hs[:,-1:,:] - self.end_h)
        for i in range(inside_hs.shape[1] - 1):
            loss += MSELoss(inside_hs[:,i:i+1,:] - inside_hs[:,i+1:i+2,:])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add here the collecting of loss values for the plots.

    def _create_changes_plot(self):

        raise NotImplementedError

    def _create_results_plot(self):

        raise NotImplementedError
