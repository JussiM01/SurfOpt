import pytest
import torch

from surfacemap import (Gaussian2D, GaussMonom, GaussPoly, Monomial,
    Polynomial, SurfaceMap)


@pytest.fixture
def monomial(constant, powers):
    params = {
        'const': constant,
        'pow': powers_vec
    }

    return Monomial(params)


@pytest.fixture
def polynomial(constants_list, powers_vec_list):
    params = {
        'const': constants_list,
        'pow': powers_vec_list
    }

    return Polynomial(params)


@pytest.fixture
def gaussian2d(mean, covariance):
    params = {
        'mean': mean,
        'cov': covariance
    }

    return Gaussian2D(params)


@pytest.fixture
def gaussmonom(mean, covariance, constant):
    params = {
        'mean': mean,
        'cov': covariance,
        'const': constant
    }

    return GaussMonom(params)


@pytest.fixture
def gausspoly(means_list, covariances_list, constants_list):
    params = {
        'mean': means_list,
        'cov': covariances_list,
        'const': constants_list
    }

    return GaussMonom(params)


@pytest.fixture
def surfacemap(poly_params, gauss_params):
    params = {
        'poly': poly_params,
        'gauss': gauss_params
    }

    return SurfaceMap(params)


# testdata0 = []

@pytest.mark.parametrize("constant,powers", testdata0)
def test_monomial(constant, powers, expected):

    pass
