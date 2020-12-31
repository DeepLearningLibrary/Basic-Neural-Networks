# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:51:51 2020

@author: Grant
"""

import random
import torch
from torch import nn, optim
import math
from IPython import display
from matplotlib import pyplot as plt
import numpy as np

def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    
def plot_data(X, y, d=0, auto=False, zoom=1):
    X = X.cpu()
    y = y.cpu()
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    plt.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    if auto is True: plt.axis('equal')
    plt.axis('off')

    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)

def plot_model(X, y, model):
    model.cpu()
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(X, y)

set_default()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 12345
random.seed(seed)
torch.manual_seed(seed)
N = 500
D = 2
C = 5

X = torch.zeros(N * C, D).to(device)
y = torch.zeros(N * C, dtype=torch.long).to(device)

for c in range(C):
    index = 0
    t = torch.linspace(0, 1, N)
    
    inner_var = torch.linspace(
            (2 * math.pi / C) * (c),
            (2 * math.pi / C) * (2 + c),
            N
    ) + torch.randn(N) * 0.2
            
    for ix in range(N * c, N * (c + 1)):
        X[ix] = t[index] * torch.FloatTensor((
                math.sin(inner_var[index]), math.cos(inner_var[index])
        ))
        y[ix] = c
        index += 1

model = nn.Sequential(
        nn.Linear(D, 500),
        nn.Tanh(),
        nn.Linear(500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.Tanh(),
        nn.Linear(500, C)
        )
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for t in range(50):
    
    #gets the result from the model
    y_pred = model(X)
    
    #compares the result to the expected output
    loss = criterion(y_pred, y)
    
    #zero the gradient before running backward pass
    optimiser.zero_grad()
    
    #Backpropagation to calculate the gradient of loss vs learnable parameters
    loss.backward()
    
    #update the parameters
    optimiser.step()

print(model)
plot_model(X, y, model)