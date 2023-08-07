import torch
import numpy as np


'''
Available Patterns:
tophat_corner
tophat_center
tophat_top
gaussian
'''

def tophat_corner(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        U0 = torch.tensor([[0.0 for _ in Y] for _ in X], dtype = torch.float, device = device)
        U0[1 : round(len(X) / 8), 1 : round(len(Y) / 8)] = value
    return U0

def tophat_center(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        U0 = torch.tensor([[0.0 for _ in Y] for _ in X], dtype = torch.float, device = device)
        U0[int(np.rint(len(X) / 2)) - int(np.rint(len(X) / 6)) : int(np.rint(len(X) / 2)) + int(np.rint(len(X) / 6)), \
            int(np.rint(len(Y) / 2)) - int(np.rint(len(Y) / 6)) : int(np.rint(len(Y) / 2)) + int(np.rint(len(Y) / 6))] = value
    return U0

def tophat_top(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        unit = round(len(X) / 15) # cut into 15 units
        U0 = torch.tensor([[0.0 for _ in Y] for _ in X], dtype = torch.float, device = device)
        #U0[1 : unit * 5, unit * 6 : unit * 8] = value
        U0[unit * 3 : unit * 5, unit * 6 : unit * 8] = value
        #U0[unit * 10 : unit * 12, unit * 6 : unit * 8] = value
        #U0[unit * 6 : unit * 8, unit * 1 : unit * 3] = value
        #U0[unit * 6 : unit * 8, unit * 11 : unit * 13] = value
    return U0

def gaussian(data_grid, value, device):
    if len(data_grid) == 2:
        X, Y = data_grid[0], data_grid[1]
        #U0 = torch.exp(torch.tensor([[(- X[i] ** 2 - Y[j] ** 2)/0.2 for j in range(len(Y))] for i in range(len(X))], \
        #    dtype = torch.float, device = device)) * value
        U0 = torch.exp(torch.tensor([[(- (X[i] / 5) ** 2 - (Y[j] / 5) ** 2) * 4 for j in range(len(Y))] for i in range(len(X))], \
            dtype = torch.float, device = device)) * value
    return U0