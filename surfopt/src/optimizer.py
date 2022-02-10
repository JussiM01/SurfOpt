import os
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
        self.regular_const = params['regular_const']
        self.plot_changes = params['plot_changes']
        self.plot_best_traj = params['plot_best']
        self.plot_results = params['plot_results']
        self.plot_all = params['plot_all']
        self.save_plots = params['save_plots']
        self.saving_name = params['saving_name']
        self.optim_type = params['optim_type']
        self.fig_params = params['fig']

    def __call__(self, surface_params, path_params):

        self._initialize(surface_params, path_params)

        for i in range(self.num_opt_steps):
            self._optim_step()
            print('optimization step {} of {}'.format(i+1, self.num_opt_steps))

        if self.plot_changes or self.plot_all:
            self._create_changes_plots()

        if self.plot_results or self.plot_all:
            self._create_results_plot()

        if self.plot_best_traj or self.plot_all:
            self._create_best_path_plot()

        best_index = self._best_indices[-1]
        optimized_path = self._paths_copies[-1][best_index,:,:]

        return optimized_path

    def _initialize(self, surface_params, path_params):

        sampler = Sampler(path_params)
        paths = sampler()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._num_trajs = paths.shape[0]
        self._num_steps = paths.shape[1]
        self._starts = paths[:,0:1,:]
        self._ends = paths[:,-1:,:]
        self._inside_trajs = torch.from_numpy(
            paths[:,1:-1,:]).to(self._device).requires_grad_(True)
        self._surfacemap = SurfaceMap(surface_params)
        self._optimizer = self._set_optimizer(self._inside_trajs)
        self._start_xys = torch.from_numpy(paths[:,0,:]).to(
            self._device)
        self._end_xys = torch.from_numpy(paths[:,-1,:]).to(self._device)
        self._start_hs = self._surfacemap(
            torch.from_numpy(paths[:,0,:]).to(self._device))
        self._end_hs = self._surfacemap(
            torch.from_numpy(paths[:,-1,:]).to(self._device))
        start_xys = torch.from_numpy(paths[:,0:1,:]).to(self._device)
        end_xys = torch.from_numpy(paths[:,-1:,:]).to(self._device)
        start_hs = torch.unsqueeze(self._surfacemap(
            torch.from_numpy(paths[:,0:1,:]).to(self._device)), dim=2)
        end_hs = torch.unsqueeze(self._surfacemap(
            torch.from_numpy(paths[:,-1:,:]).to(self._device)), dim=2)
        self._start_points = torch.cat([start_xys, start_hs], dim=2)
        self._end_points = torch.cat([end_xys, end_hs], dim=2)
        self._grid = create_grid(self.fig_params['grid'], self._surfacemap)
        self._loss_copies = {'losses': [], 'mean_losses': []}
        self._best_indices = []
        self._paths_copies = []

    def _set_optimizer(self, inside_trajs):

        if self.optim_type == 'Adam':
            return torch.optim.Adam([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'AdamW':
            return torch.optim.AdamW([inside_trajs], lr=self.learning_rate)

        elif self.optim_type == 'SGD':
            return torch.optim.SGD([inside_trajs], lr=self.learning_rate)

    def _optim_step(self):

        self._copy_trajs()
        in_hs = torch.unsqueeze(self._surfacemap(self._inside_trajs), dim=2)
        in_points = torch.cat([self._inside_trajs, in_hs], dim=2)
        no_ends = torch.cat([self._start_points, in_points], dim=1)
        no_starts =  torch.cat([in_points, self._end_points], dim=1)

        loss = torch.zeros(self._num_trajs).to(self._device)
        ds_squares = torch.sum((no_ends -no_starts)**2, dim=2)
        meand_ds2 = torch.mean(ds_squares, dim=1)
        loss += meand_ds2
        meand_ds2 = torch.stack((self._num_steps -1)*[meand_ds2], dim=1)
        reg_loss = torch.mean((meand_ds2 - ds_squares)**2, dim=1)
        loss += self.regular_const * reg_loss

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
        self._paths_copies.append(trajs)

    def _copy_losses(self, loss, mean_loss):

        self._best_indices.append(torch.argmin(loss).item())
        self._loss_copies['losses'].append(loss.tolist())
        self._loss_copies['mean_losses'].append(mean_loss.item())

    def _plot_or_save(self, plot_type):

        if self.save_plots:
            dir = os.path.join('surfopt', 'plots', self.saving_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            filename = plot_type + '_' + '.png'
            plt.savefig(os.path.join(dir, filename))

        else:
            plt.show()

    def _create_changes_plots(self):

        colormap = cm.get_cmap('inferno_r', self.num_opt_steps)
        for i in range(self._num_trajs):
            fig, ax = create_plot(self.fig_params['plot'])
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            X, Y, Z = self._grid
            ax.contour(X, Y, Z, colors='lightgray')
            for j in range(self.num_opt_steps):
                xs = self._paths_copies[j][i,:,0]
                ys = self._paths_copies[j][i,:,1]
                ax.plot(xs, ys, color=colormap.colors[j])
            ax.set_title('Optimization of the path {}'.format(i+1))

            self._plot_or_save('changes_plot_traj{}'.format(i+1))

    def _create_best_path_plot(self):

        fig, ax = create_plot(self.fig_params['plot'])
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        X, Y, Z = self._grid
        ax.contour(X, Y, Z, colors='lightgray')
        best_index = self._best_indices[-1]
        xs = self._paths_copies[-1][best_index,:,0]
        ys = self._paths_copies[-1][best_index,:,1]
        ax.plot(xs, ys, color='k')
        ax.set_title('Best path after optimization')

        self._plot_or_save('best_path_plot')

    def _create_results_plot(self):

        params = deepcopy(self.fig_params)
        params['bound'] = None
        fig, ax = create_plot(params)
        xs = [j for j in range(self.num_opt_steps)]
        for i in range(self._num_trajs):
            label = 'path {} loss'.format(i+1)
            ys = [self._loss_copies['losses'][j][i]
                  for j in range(self.num_opt_steps)]
            ax.plot(xs, ys, label=label)
        ys = [self._loss_copies['mean_losses'][j]
              for j in range(self.num_opt_steps)]
        ax.plot(xs, ys, color='k', linestyle='--', label='Mean loss')
        ax.set_xlabel('Optimization step')
        ax.set_ylabel('Loss')
        ax.set_title('Losses of the paths and their mean loss')
        if self._num_trajs <= 10:
            ax.legend()

        self._plot_or_save('results_plot')
