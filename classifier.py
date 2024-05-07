import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from model import Classifier

batch_size = 32
lr = 0.001
epochs = 500
finetune_rate = 0.1

model_file = "output/hhar_pretrain_autoencoder/model.pt"

def evaluate(model, dataloader, criterion, device):
  model.eval()  # Set the model to evaluation mode
  
  results = []  # prediction results
  labels = []
  total_loss = 0
  with torch.no_grad():  # No need to track gradients
    for data, target in dataloader:
      data = data.to(device)
      target = target.to(device)
      logits, outputs = model(data)
      loss = criterion(logits, target)
      total_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      results.append(predicted)
      labels.append(target)
  
  average_loss = total_loss / len(dataloader)  
  acc, f1 = utils.stat_acc_f1(torch.cat(labels, 0).cpu().numpy(), torch.cat(results, 0).cpu().numpy())

  return average_loss, acc, f1

if __name__ == "__main__":
  # Load the dataset
  data, label = utils.load_data('data/hhar')
  # splitting the dataset
  utils.set_seeds(2405022300)  # must be the same as the one in train.py
  train_data, train_label, vali_data, vali_label, test_data, test_label = utils.split_data(
      data, label, train_rate=0.8, dev_rate=0.1)
  train_data, train_label = utils.finetune_data(train_data, train_label, train_rate=finetune_rate)
  vali_data, vali_label = utils.unique_label(vali_data, vali_label)
  test_data, test_label = utils.unique_label(test_data, test_label)
  
  dataset_train = utils.IMUDataset(train_data, train_label)
  dataset_vali = utils.IMUDataset(vali_data, vali_label)
  dataset_test = utils.IMUDataset(test_data, test_label)
  dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  dataloader_vali = DataLoader(dataset_vali, batch_size=batch_size, shuffle=False)
  dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
  
  # Load the pre-trained encoder weights
  classifier = Classifier()

  # Setup for classifier training
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(classifier.parameters(), lr=lr)
  device = utils.get_device("0")

  # Training loop for the classifier
  def train_classifier(model, dataloader_train, dataloader_vali, dataloader_test, optimizer, criterion, device, model_file, epochs=10):
    model.to(device)
    model.load_pretrain(model_file, device)
    
    vali_acc_best = 0.0
    best_stat = None
    for epoch in range(epochs):
      model.train()
      loss_sum = 0.0  # the sum of iteration losses to get average loss in every epoch
      time_sum = 0.0
      for data, target in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs}"):
        data = data.to(device)
        target = target.to(device)
        start_time = time.time()
        
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = criterion(logits, target)
        
        loss.backward()
        optimizer.step()
        time_sum += time.time() - start_time
        loss_sum += loss.item()
      
      train_loss, train_acc, train_f1 = evaluate(model, dataloader_train, criterion, device)
      vali_loss, vali_acc, vali_f1 = evaluate(model, dataloader_vali, criterion, device)
      test_loss, test_acc, test_f1 = evaluate(model, dataloader_test, criterion, device)
      
      # print
      print(f"Epoch {epoch+1}/{epochs} ({time_sum:.2f}s):")
      print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}")
      print(f"  Vali loss: {vali_loss:.4f}, acc: {vali_acc:.4f}, f1: {vali_f1:.4f}")
      print(f"  Test loss: {test_loss:.4f}, acc: {test_acc:.4f}, f1: {test_f1:.4f}")
      
      if vali_acc > vali_acc_best:
        vali_acc_best = vali_acc
        best_stat = (train_loss, vali_loss, test_loss, train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
    
    print('The Total Epoch have been reached.')
    print('Loss:  %0.3f/%0.3f/%0.3f\nBest Accuracy: %0.3f/%0.3f/%0.3f\nF1: %0.3f/%0.3f/%0.3f' % best_stat)

  # Example training call
  train_classifier(classifier, dataloader_train, dataloader_vali, dataloader_test, optimizer, criterion, device, model_file, epochs)