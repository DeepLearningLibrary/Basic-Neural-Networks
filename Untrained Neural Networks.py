# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:57:05 2020

@author: Grant
"""

import torch
import torch.nn as nn
from matplotlib.pyplot import plot, title, axis
from matplotlib import pyplot as plt

def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    
def show_scatterplot(X, colors, title=''):
    colors = colors.cpu().numpy()
    X = X.cpu().numpy()
    plt.figure()
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=30)
    # plt.grid(True)
    plt.title(title)
    plt.axis('off')

set_default()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_points = 100
X = torch.randn(n_points, 5).to(device)
colors = X[:, 0]
"""
#layere Neural Network Model 1
model = nn.Sequential(
        nn.Linear(5, 5, bias=False), #Layer 1
        nn.Linear(5, 5, bias=False), #Layer 2
        nn.Tanh()) #Layer 3

model.to(device)

#Layer 2 reverses the weights multiplication of Layer 1
for s in range(1, 15):
    W_1 = s * torch.eye(5) #Prepare the weights for layer 1
    W_2 = 1/s * torch.eye(5) #Prepare the weights for layer 2
    model[0].weight.data.copy_(W_1) #set the weights for Layer 1
    model[1].weight.data.copy_(W_2) #set the weights for Layer 2
    Y = model(X).data #Pass data through the layers, goes through Layers 1, 2, 3
    
    show_scatterplot(Y, colors, title=f'f(x), s={s}')
"""





#layere Neural Network Model 2
model = nn.Sequential(
        nn.Linear(5, 10, bias=False), #Layer 1
        nn.Tanh(), #Layer 2
        nn.Linear(10, 5, bias=False), #Layer 3
        nn.Tanh(), #layer 4
        nn.Linear(5, 10, bias=False), #Layer 5
        nn.Tanh(), #layer 6
        nn.Linear(10, 5, bias=False), #Layer 7
        nn.Tanh()) #Layer 8

model.to(device)

for s in range(1, 15):
    W_1 = s * torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    model[0].weight.data.copy_(W_1) #set the weights for Layer 1
    
    W_2 = 1/s * torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    model[2].weight.data.copy_(W_2) #set the weights for Layer 3
    
    W_3 = s * torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    model[4].weight.data.copy_(W_3) #set the weights for Layer 5
    
    W_4 = 1/s * torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    model[6].weight.data.copy_(W_4) #set the weights for Layer 7
    
    Y = model(X).data #Pass data through the layers, goes through Layers 1, 2, 3, 4, 5, 6, 7, 8
    
    show_scatterplot(Y, colors, title=f'f(x), s={s}')
    