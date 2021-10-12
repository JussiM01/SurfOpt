import torch
import torch.nn as nn


class Mononomial(nn.Module):

    def __init__(self, params):

        super(Polynomial, self).__init__()
        self.params = params

    def forward(self, tensor):

        x_values = (self.params['const'][0] *
                    torch.pow(tensor[:, 0], self.params['power'][0]))
        y_values = (self.params['const'][1] *
                    torch.pow(tensor[:, 1], self.params['power'][1]))

        return torch.stack([x_values, y_values], axis=1)


class Polynomial(nn.Module):

    def __init__(self, params):

        super(Polynomial, self).__init__()
        self.monomials = [Mononomial(params[i]) for i range(len(params)))]

    def forward(self, tensor):

        value = torch.zeros_like(tensor)
        for func in self.monomials:
            value += func(tensor)

        return value


class Gaussian2D(nn.Module):

    def __init__(self, params):
        super(Gaussian2D, self).__init__()
        self.mean = torch.Tensor(params['mean'])
        self.cov_matrix = torch.Tensor(params['cov'])
        self.const = torch.Tensor(params['const'])

    def forward(self, point):

        point = point -self.mean
        value = torch.dot(point, torch.matmul(self.cov_matrix, point))

        return self.const*torch.exp(-value)


class GaussMonom(nn.Module):

    def __init__(self, params):

        super(GaussMonom, self).__init__()
        self.func = Gaussian2D(params)

    def forward(self, tensor):

        zetas = [self.func(t) for t in torch.unbind(tensor, dim=0)]

        return torch.stack(zetas, axis=0)



class GaussPoly(nn.Module):

    def __init__(self, params):

        super(GaussPoly, self).__init__()
        self.gaussmonoms = [GaussMonom(params[i]) for i range(len(params)))]

    def forward(self, tensor):

        value = torch.zeros_like(tensor)
        for func in self.gaussmonoms:
            value += func(tensor)

        return value


class SurfaceMap(nn.Module):

    def __init__(self, params):

        self.functions = []
        if 'poly' in params:
            self.functions.append(Polynomial(params['poly']))
        if 'gauss' in params:
            self.functions.append(GaussPoly(params['gauss']))

    def forward(self, plane_points):

        zetas = torch.zeros_like(plane_points[:, 0])
        for func in self.functions:
            zetas += func(plane_points)

        return zetas
