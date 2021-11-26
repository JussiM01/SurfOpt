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

        if self.plot_changes:
            self._create_changes_plots()

        if self.plot_results:
            self._create_results_plot()

        # Return here the trajectory which has the lowest lenght
        # after the optimization.

    def _initialize(self, surface_params, trajectory_params):

        sampler = Sampler(trajectory_params)
        trajectories = sampler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._inside_trajs = torch.from_numpy(
            trajectories[:,1:-1,:]).to(device).requires_grad_(True)
        self._surfacemap = SurfaceMap(surface_params)
        self._optimizer = self._set_optimizer(self._inside_trajs)
        self._start_h = self._surfacemap(torch.from_numpy(trajectories[:,0,:))
        self._end_h = self._surfacemap(torch.from_numpy(trajectories[:,-1,:]))
        self._trajs_copies = []
        self._loss_copies = []
        self._best_indices = []

    def _set_optimizer(self, inside_trajs):

        if self.optim_type == 'Adam':
            return torch.optim.Adam([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'AdamW':
            return torch.optim.AdamW([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'SGD':
            return torch.optim.SGD([inside_trajs], lr=self.learning_rate)

    def _optim_step(self):

        self._copy_trajs()
        inside_hs = self._surfacemap(self._inside_trajs)
        loss = torch.zeros_like(inside_hs[:,0,:])
        loss += torch.sum((self._start_h - inside_hs[:,0,:])**2, dim=1)
        loss += torch.sum((inside_hs[:,-1,:] - self._end_h)**2, dim=1)
        for i in range(inside_hs.shape[1] - 1):
            loss += torch.sum(
                (inside_hs[:,i,:] - inside_hs[:,i+1,:])**2, dim=1)

        self._optimizer.zero_grad()
        mean_loss = torch.mean(loss)
        mean_loss.backward()
        self._optimizer.step()
        self._copy_losses(loss)

    def _copy_trajs(self):
        # Saves a copy of the current trajectories (with end points) as a list.
        raise NotImplementedError

    def _copy_losses(self, loss):
        # Saves a copy of the current loss and best index (list & float).
        raise NotImplementedError

    def _create_changes_plots(self):

        raise NotImplementedError

    def _create_results_plot(self):

        raise NotImplementedError
