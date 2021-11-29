import torch

from copy import deepcopy
from sampler import Sampler
from surfacemap import SurfaceMap
from utils import create_grid, create_plot


class Optimizer:

    def __init__(self, params):

        self.num_steps = params['num_steps']
        self.learning_rate = params['learning_rate']
        self.plot_changes = params['plot_changes']
        self.plot_results = params['plot_results']
        self.save_plots = params['save_plots']
        self.optim_type = params['optim_type']
        self.fig_params = params['fig']

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
        self._starts = torch.from_numpy(trajectories[:,0:1,:])
        self._ends = torch.from_numpy(trajectories[:,-1:,:])
        self._inside_trajs = torch.from_numpy(
            trajectories[:,1:-1,:]).to(device).requires_grad_(True)
        self._surfacemap = SurfaceMap(surface_params)
        self._optimizer = self._set_optimizer(self._inside_trajs)
        self._start_hs = self._surfacemap(torch.from_numpy(trajectories[:,0,:))
        self._end_hs = self._surfacemap(torch.from_numpy(trajectories[:,-1,:]))
        self._grid = create_grid(self.fix_params['grid'], self._surfacemap)
        self._loss_copies = {'losses': [], 'mean_losses': []}
        self._best_indices = []
        self._trajs_copies = []

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
        loss += torch.sum((self._start_hs - inside_hs[:,0,:])**2, dim=1)
        loss += torch.sum((inside_hs[:,-1,:] - self._end_hs)**2, dim=1)
        for i in range(inside_hs.shape[1] - 1):
            loss += torch.sum(
                (inside_hs[:,i,:] - inside_hs[:,i+1,:])**2, dim=1)

        self._optimizer.zero_grad()
        mean_loss = torch.mean(loss)
        mean_loss.backward()
        self._optimizer.step()
        self._copy_losses(loss, mean_loss)

    def _copy_trajs(self):

        inside_trajs = deepcopy(self._inside_trajs)
        inside_trajs = inside_trajs.cpu().detach()
        trajs = torch.cat([self._starts, inside_trajs, self._ends], dim=1)
        self._trajs_copies.append(trajs.tolist())

    def _copy_losses(self, loss, mean_loss):

        self._best_indices.append(torch.argmin(loss).item())
        self._loss_copies['losses'].append(loss.tolist())
        self._loss_copies['mean_losses'].append(mean_loss.item())

    def _create_changes_plots(self):

        raise NotImplementedError

    def _create_results_plot(self):

        raise NotImplementedError
