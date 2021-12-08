import matplotlib.pyplot as plt
import numpy as np
import torch

from copy import deepcopy
from matplotlib import cm
from src.sampler import Sampler
from src.surfacemap import SurfaceMap
from src.utils import create_grid, create_plot


class Optimizer:

    def __init__(self, params):

        self.num_opt_steps = params['num_opt_steps']
        self.learning_rate = params['learning_rate']
        self.plot_changes = params['plot_changes']
        self.plot_best_traj = params['plot_best']
        self.plot_results = params['plot_results']
        self.save_plots = params['save_plots']
        self.optim_type = params['optim_type']
        self.fig_params = params['fig']

    def __call__(self, surface_params, trajectory_params):

        self._initialize(surface_params, trajectory_params)

        for i in range(self.num_opt_steps):
            self._optim_step()
            print('optimization step {} of {}'.format(i+1, self.num_opt_steps))

        if self.plot_changes:
            self._create_changes_plots()

        if self.plot_best_traj:
            self._create_best_traj_plot()

        if self.plot_results:
            self._create_results_plot()

        best_index = self._best_indices[-1]
        optimized_trajectory = self._trajs_copies[-1][best_index,:,:]

        return optimized_trajectory

    def _initialize(self, surface_params, trajectory_params):

        sampler = Sampler(trajectory_params)
        trajectories = sampler()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_trajs = trajectories.shape[0]
        self._starts = trajectories[:,0:1,:]
        self._ends = trajectories[:,-1:,:]
        self._inside_trajs = torch.from_numpy(
            trajectories[:,1:-1,:]).to(device).requires_grad_(True)
        self._surfacemap = SurfaceMap(surface_params)
        self._optimizer = self._set_optimizer(self._inside_trajs)
        self._start_xys = torch.from_numpy(trajectories[:,0,:]).to(device)
        self._end_xys = torch.from_numpy(trajectories[:,-1,:]).to(device)
        self._start_hs = self._surfacemap(
            torch.from_numpy(trajectories[:,0,:]).to(device))
        self._end_hs = self._surfacemap(
            torch.from_numpy(trajectories[:,-1,:]).to(device))
        self._grid = create_grid(self.fig_params['grid'], self._surfacemap)
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
        loss = torch.zeros_like(inside_hs[:,0])
        loss += torch.sum(
            (self._start_xys - self._inside_trajs[:,0,:])**2, dim=1)
        loss += torch.sum(
            (self._inside_trajs[:,-1,:] - self._end_xys)**2, dim=1)
        loss += (self._start_hs - inside_hs[:,0])**2
        loss += (inside_hs[:,-1] - self._end_hs)**2
        for i in range(inside_hs.shape[1] - 1):
            loss += torch.sum((self._inside_trajs[:,i,:]
                              - self._inside_trajs[:,i+1,:])**2, dim=1)
            loss += (inside_hs[:,i] - inside_hs[:,i+1])**2

        self._optimizer.zero_grad()
        mean_loss = torch.mean(loss)
        mean_loss.backward()
        self._optimizer.step()
        self._copy_losses(loss, mean_loss)

    def _copy_trajs(self):

        inside_trajs = deepcopy(self._inside_trajs)
        inside_trajs = inside_trajs.cpu().detach().numpy()
        starts = deepcopy(self._starts)
        ends = deepcopy(self._ends)
        trajs = np.concatenate([starts, inside_trajs, ends], axis=1)
        self._trajs_copies.append(trajs)

    def _copy_losses(self, loss, mean_loss):

        self._best_indices.append(torch.argmin(loss).item())
        self._loss_copies['losses'].append(loss.tolist())
        self._loss_copies['mean_losses'].append(mean_loss.item())

    def _create_changes_plots(self):

        colormap = cm.get_cmap('inferno', self.num_opt_steps)
        for i in range(self._num_trajs):
            fig, ax = create_plot(self.fig_params['plot'])
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            X, Y, Z = self._grid
            ax.contour(X, Y, Z, colors='lightgray')
            for j in range(self.num_opt_steps):
                label = 'Opt. step {}'.format(j+1)
                xs = self._trajs_copies[j][i,:,0]
                ys = self._trajs_copies[j][i,:,1]
                ax.plot(xs, ys, color=colormap.colors[j], label=label)
            ax.set_title('Optimization steps of the trajectory {}'.format(i+1))
            ax.legend()
            plt.show()

    def _create_best_traj_plot(self):

        fig, ax = create_plot(self.fig_params['plot'])
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        X, Y, Z = self._grid
        ax.contour(X, Y, Z, colors='lightgray')
        best_index = self._best_indices[-1]
        xs = self._trajs_copies[-1][best_index,:,0]
        ys = self._trajs_copies[-1][best_index,:,1]
        ax.plot(xs, ys, color='k')
        ax.set_title('Best trajectory after optimization')
        plt.show()

    def _create_results_plot(self):

        params = deepcopy(self.fig_params)
        params['bound'] = None
        fig, ax = create_plot(params)
        colormap = cm.get_cmap('viridis', self.num_opt_steps)
        xs = [j for j in range(self.num_opt_steps)]
        for i in range(self._num_trajs):
            label = 'Trajectory {} losses'.format(i+1)
            ys = [self._loss_copies['losses'][j][i]
                  for j in range(self.num_opt_steps)]
            ax.plot(xs, ys, color=colormap.colors[i], label=label)
        ys = [self._loss_copies['mean_losses'][j]
              for j in range(self.num_opt_steps)]
        ax.plot(xs, ys, color='r', label='Mean losses')
        ax.set_xlabel('Optimization step')
        ax.set_ylabel('Loss')
        ax.set_title('Losses')
        ax.legend()
        plt.show()
