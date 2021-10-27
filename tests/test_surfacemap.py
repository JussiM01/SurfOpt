import pytest
import torch

from src.surfacemap import GaussMonom, Monomial, SurfaceMap


device = torch.device("cuda" if torch.cuda.is_available()
    else "cpu")

testdata0 = [
    [100.0, [2, 3], 10800.0],
    [10.0, [3, 4], -6480.0],
    [0.1, [4, 5], 388.8]
]

tensor0 = torch.Tensor([[-2.0, 3.0]]).to(device)


@pytest.mark.parametrize("constant,powers,expected", testdata0)
def test_monomial(constant, powers, expected):
    model = Monomial({'const': constant, 'pow': powers})
    result = model(tensor0)

    assert expected == result


testdata1 = [
    [[-1.0, 1.0], torch.exp(torch.Tensor([0.0, -2.0, -8.0]))],
    [[0.0, 0.0], torch.exp(torch.Tensor([-2.0, 0.0, -2.0]))],
    [[1.0, -1.0], torch.exp(torch.Tensor([-8.0, -2.0, 0.0]))]
]

tensor1 = torch.Tensor([[-1.0, 1.0] , [0.0, 0.0], [1.0, -1.0]])


@pytest.mark.parametrize("mean,expected", testdata1)
def test_gaussmonom(mean, expected):
    params = {'mean': mean, 'cov': [[1.0, 0.0], [0.0, 1.0]], 'const': 1.0}
    model = GaussMonom(params)
    result = model(tensor1)

    assert torch.equal(expected, result)


testdata2 = [
    [torch.Tensor([[-3.0, -1.0], [-2.0, -1.0], [1.0, -1.0], [-3.0, 1.0],
                  [-2.0, 1.0], [0.0, 1.0], [1.0, 1.0]]),
     {'poly': {'const': [1.0, 3.0], 'pow': [[3, 1], [2, 1]]}},
     torch.Tensor([0.0, -4.0, 0.0, -4.0, 0.0, 4.0, 0.0, 4.0])
    ],
    [torch.Tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]),
     {'gauss': {
                'mean': [[-1.0, 1.0], [1.0, -1.0]],
                'cov': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                'const': [1.0, 1.0]}
                },
     (torch.exp(torch.Tensor([0.0, -2.0, -8.0]))
        + torch.exp(torch.Tensor([-8.0, -2.0, 0.0]))
     )
    ],
    [torch.Tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [-1.0, -1.0],
                   [1.0, 1.0]]),
     {
      'poly': {'const': [1.0, 1.0], 'pow': [[1, 0], [0, 1]]},
      'gauss': {
                'mean': [[-1.0, 1.0], [1.0, -1.0]],
                'cov': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                'const': [1.0, 1.0]
                }
     },
     (torch.exp(torch.Tensor([0.0, -2.0, -8.0, -4.0, -4.0]))
        + torch.exp(torch.tensor([-8.0, -2.0, 0.0, -4.0, -4.0]))
        + torch.Tensor([0.0, 0.0, 0.0, 2.0, -2.0])
     )
    ]
]


@pytest.mark.parametrize("tensor,params,expected", testdata2)
def test_surfacemap(tensor, params, expected):
    model = SurfaceMap(params)
    result = model(tensor)

    assert torch.equal(expected, result)
