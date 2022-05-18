# load model

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from main import *

import utils

import os
import numpy as np
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

from matplotlib import pyplot as plt

%%capture
model_path = f'./modelscopy/mnist_hard_model_64_0.0_0.0_10.pth'
model = Model().to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model.eval()


def visualize_plots(model,method,batch_size,tau_plus=None,beta=None,dataset_name = 'mnist'):


  if method == 'adv' or method == 'easy':
    tau_plus = 0
    beta = 0

  if method == 'adv':

    train_data, _, test_data = utils.get_dataset(dataset_name, root='../data/', pair=True)
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    print(batch_size)
    
    coords1 = []
    coords2 = []
    labels = []
    Ngs = []
    Pos = []

    data_loader = test_loader

    for _ in range(5):
      for data1, data2, target in tqdm(data_loader):

          if data1.shape[0]!=batch_size:
            break

          data1 = data1.to(device, non_blocking=True)
          data2 = make_adv(model, data1,tau_plus,beta,batch_size)

          with torch.no_grad():
            
              target = target.to(device,non_blocking=True)

              _, out1 = model(data1)
              _, out2 = model(data2)

              labels.append(target.cpu().numpy())
              normalized_output1 = out1.cpu().detach().numpy() / np.linalg.norm(out1.cpu().detach().numpy(), axis=1, keepdims=True)
              normalized_output2 = out2.cpu().detach().numpy() / np.linalg.norm(out2.cpu().detach().numpy(), axis=1, keepdims=True)

              coords1.append(normalized_output1)
              coords2.append(normalized_output2)
              _, ng,pos = criterion(out1, out2, target, target, tau_plus, batch_size, beta, 'hard', epoch=10, temp=.5, return_Ng=True)
              Ngs.append(ng)
              Pos.append(pos)

  if method == 'hard' or method == 'easy':

    train_data, _, test_data = utils.get_dataset(dataset_name, root='../data/', pair=True)
    test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    coords1 = []
    coords2 = []
    labels = []
    Ngs = []
    Pos = []

    data_loader = test_loader
    
    for _ in range(5):
      for data1, data2, target in tqdm(data_loader):

          if data1.shape[0]!=batch_size:
            break

          data1 = data1.to(device, non_blocking=True)
          data2 = data2.to(device, non_blocking=True)

          with torch.no_grad():
            
              target = target.to(device,non_blocking=True)

              _, out1 = model(data1)
              _, out2 = model(data2)

              labels.append(target.cpu().numpy())
              normalized_output1 = out1.cpu().detach().numpy() / np.linalg.norm(out1.cpu().detach().numpy(), axis=1, keepdims=True)
              normalized_output2 = out2.cpu().detach().numpy() / np.linalg.norm(out2.cpu().detach().numpy(), axis=1, keepdims=True)

              coords1.append(normalized_output1)
              coords2.append(normalized_output2)
              _, ng,pos = criterion(out1, out2, target, target, tau_plus, batch_size, beta, method, epoch=10, temp=.5, return_Ng=True)
              Ngs.append(ng)
              Pos.append(pos)

  Ngs = torch.cat(Ngs).detach().cpu().numpy()
  Pos = torch.cat(Pos).detach().cpu().numpy()

  plt.hist(Pos/(Pos+Ngs),bins = 100, density = True)
  plt.show()

  return Ngs