import torch
import torch.nn as nn


class Monomial(nn.Module):

    def __init__(self, params):

        super(Monomial, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.params = params

    def forward(self, tensor):

        zetas = (self.params['const']
                 * torch.pow(tensor[:, 0], self.params['pow'][0])
                 * torch.pow(tensor[:, 1], self.params['pow'][1]))

        return zetas


class Polynomial(nn.Module):

    def __init__(self, params):

        super(Polynomial, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.monomials = []
        for i in range(len(params['const'])):
            sub_params = {}
            for key in params:
                sub_params[key] = params[key][i]
            self.monomials.append(Monomial(sub_params))

    def forward(self, tensor):

        zetas = torch.zeros_like(tensor[:, 0])
        for func in self.monomials:
            zetas += func(tensor)

        return zetas


class Gaussian2D(nn.Module):

    def __init__(self, params):
        super(Gaussian2D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.mean = torch.Tensor(params['mean'])
        self.cov_matrix = torch.Tensor(params['cov'])
        self.const = torch.Tensor([params['const']])

    def forward(self, point):

        point = point -self.mean
        zeta = torch.dot(point, torch.matmul(self.cov_matrix, point))

        return self.const*torch.exp(-zeta)


class GaussMonom(nn.Module):

    def __init__(self, params):

        super(GaussMonom, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.func = Gaussian2D(params)

    def forward(self, tensor):

        zetas = [self.func(t) for t in torch.unbind(tensor, dim=0)]

        return torch.cat(zetas, axis=0)



class GaussPoly(nn.Module):

    def __init__(self, params):

        super(GaussPoly, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.gaussmonoms = []
        for i in range(len(params['mean'])):
            sub_params = {}
            for key in params:
                sub_params[key] = params[key][i]
            self.gaussmonoms.append(GaussMonom(sub_params))

    def forward(self, tensor):

        zetas = torch.zeros_like(tensor[:, 0])
        for func in self.gaussmonoms:
            zetas += func(tensor)

        return zetas


class SurfaceMap(nn.Module):

    def __init__(self, params):

        super(SurfaceMap, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
            else "cpu")
        self.functions = []
        if 'poly' in params:
            self.functions.append(Polynomial(params['poly']))
        if 'gauss' in params:
            self.functions.append(GaussPoly(params['gauss']))

    def forward(self, plane_points):

        shape = plane_points.shape

        if len(shape) == 2:
            zetas = torch.zeros_like(plane_points[:, 0])
            for func in self.functions:
                zetas += func(plane_points)

        elif len(shape) == 3:
            all_zetas = []
            for i in range(shape[0]):
                zeta_seq = torch.zeros_like(plane_points[0, :, 0])
                for func in self.functions:
                    zeta_seq += func(plane_points)
                all_zetas.append(zeta_seq)
            zetas = torch.stack(all_zetas, axis=0)

        return zetas
