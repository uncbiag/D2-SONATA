import torch

'''
Available Patterns:
constant
gaussian
'''

def constant(data_grid, value, device):
    if len(data_grid) == 2:
        out = torch.ones([len(data_grid[0]), len(data_grid[1])], dtype = torch.float, device = device) * value
    else:
        raise NotImplementedError
    return out

def gaussian(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        out = torch.exp(torch.tensor([[- X[i] ** 2 - Y[j] ** 2 for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device)) * value
    else:
        raise NotImplementedError
    return out

def stroke(data_grid, value, device):
    if len(data_grid) == 2:
        out = gaussian(data_grid, value, device)
        X, Y = data_grid[0], data_grid[1]
        for i in range(int(len(X) / 8), int(3 * len(X) / 8)):
            for j in range(int(3 * len(Y) / 8), int(5 * len(Y) / 8)):
                out[i, j] = torch.exp(torch.tensor(- X[i] ** 2 - Y[j] ** 2, dtype = torch.float, device = device)) * value * 0.5
    else:
        raise NotImplementedError
    return out