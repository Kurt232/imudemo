import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Autoencoder

batch_size = 32
lr = 0.001
epochs = 100

model_name = "autoencoder"
save_path = "/data/hhar_pretrain_" + model_name + "/"


def evaluate_autoencoder(model, dataloader, criterion, device):
  model.eval()  # Set the model to evaluation mode
  total_loss = 0
  with torch.no_grad():  # No need to track gradients
    for data, target in dataloader:
      data = data.to(device)
      outputs, _ = model(data)
      loss = criterion(outputs, target)
      total_loss += loss.item()
  average_loss = total_loss / len(dataloader)
  return average_loss


if __name__ == '__main__':
  # Load the dataset
  data, label = utils.load_data('data/hhar')
  # splitting the dataset
  utils.set_seeds(2405022300)  # must be the same as the one in train.py
  train_data, train_label, vali_data, vali_label, test_data, test_label = utils.split_data(
      data, label, train_rate=0.8, dev_rate=0.1)

  # Data Loading
  dataset_train = utils.Dataset4Pretrain(train_data)
  dataset_test = utils.Dataset4Pretrain(test_data)
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
  # Define model, loss function, and optimizer
  autoencoder = Autoencoder()
  criterion = nn.MSELoss()
  optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
  device = utils.get_device("0")

  # Training loop for the autoencoder
  def train_autoencoder(model, dataloader_train, dataloader_test, optimizer, criterion, device, epochs):
    model.to(device)
    best_loss = 1e3
    
    for epoch in range(epochs):
      loss_sum = 0. 
      time_sum = 0.
      model.train()
      for data, target in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}"):
        data = data.to(device)
        start_time = time.time()
        optimizer.zero_grad()
        outputs, _ = model(data)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        time_sum += time.time() - start_time
        loss_sum += loss.item()
        
      loss_eva = evaluate_autoencoder(model, dataloader_test, criterion)
      print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (epoch + 1, epochs, loss_sum / len(dataloader_train), loss_eva))
      print("Train execution time: %.5f seconds" % (time_sum / len(dataloader_train)))
      if loss_eva < best_loss:
          best_loss = loss_eva
          torch.save(model.state_dict(), save_path +  f'model.pt')
          
    print('The Total Epoch have been reached.')
    print('Loss: %0.3f' % best_loss)

  train_autoencoder(autoencoder, dataloader_train, dataloader_test, optimizer, criterion, device, epochs)
