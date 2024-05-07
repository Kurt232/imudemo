import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):

  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(120 * 6, 128),
      nn.ReLU(), 
      nn.Linear(128, 64))
    self.decoder = nn.Sequential(
      nn.Linear(64, 128), 
      nn.ReLU(),
      nn.Linear(128, 120 * 6))

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded, encoded


class Classifier(nn.Module):

  def __init__(self, num_classes=6):
    super(Classifier, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(120 * 6, 128),
      nn.ReLU(), 
      nn.Linear(128, 64))
    self.classifier = nn.Sequential(
        nn.Linear(64, 32),  
        nn.ReLU(),
        nn.Linear(32, num_classes) 
    )

  def forward(self, x):
    h = self.encoder(x)
    h = self.classifier(h)
    return F.softmax(h, dim=1), h
  
