import numpy as np

from scipy.stats import multivariate_normal
from src.utils import write_dict


def create(params):

    means, covs, constants = random_gaussians(params)
    data = {'gauss': {'mean': means, 'cov': covs, 'const': constants}}
    filename = ('gauss' + str(params['num_gauss']) + 'seed'
        + str(params['random_seed']) +  '.json')

    write_dict(data, filename, 'config_files')


def random_gaussians(params):

    pos_means, pos_covs = sample_gaussians(params)
    neg_means, neg_covs = sample_gaussians(params)

    pos = []
    neg = []

    for i in range(params['num_gauss']):
        rv_pos = multivariate_normal(pos_means[i], pos_covs[i])
        rv_neg = multivariate_normal(neg_means[i], neg_covs[i])
        pos_peak_value = rv_pos.pdf(pos_means[i])
        neg_peak_value = -1*rv_neg.pdf(neg_means[i])

        pos.append(params['scale']*pos_peak_value)
        neg.append(params['scale']*neg_peak_value)

    means = pos_means + neg_means
    covs = pos_covs + neg_covs
    constants = pos + neg

    return means, covs, constants


def sample_gaussians(params):

    means = []
    covs = []

    for _ in range(params['num_gauss']):
        mean_x = np.random.uniform(params['x_min'], params['x_max'])
        mean_y = np.random.uniform(params['y_min'], params['y_max'])

        cov_xx = np.random.uniform(params['diag_min'], params['diag_max'])
        cov_xy = np.random.uniform(params['offd_min'], params['offd_max'])
        cov_yy = np.random.uniform(params['diag_min'], params['diag_max'])

        means.append([mean_x, mean_y])
        covs.append([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    return means, covs
