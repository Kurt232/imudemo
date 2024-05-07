import random
import numpy as np
import torch
import sys
import os

from torch.utils.data import Dataset
from sklearn.metrics import f1_score

def set_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def get_device(gpu):
  if gpu is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device(
        "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  print("%s (%d GPUs)" % (device, n_gpu))
  return device


def split_last(x, shape):
  "split the last dimension to given shape"
  shape = list(shape)
  assert shape.count(-1) <= 1
  if -1 in shape:
    shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
  return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
  "merge the last n_dims to a dimension"
  s = x.size()
  assert n_dims > 1 and n_dims < len(s)
  return x.view(*s[:-n_dims], -1)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
  ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
  pvals = p * np.power(1 - p, np.arange(max_gram))
  # alpha = 6
  # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
  pvals /= pvals.sum(keepdims=True)
  mask_pos = set()
  while len(mask_pos) < goal_num_predict:
    n = np.random.choice(ngrams, p=pvals)
    n = min(n, goal_num_predict - len(mask_pos))
    anchor = np.random.randint(seq_len)
    if anchor in mask_pos:
      continue
    for i in range(anchor, min(anchor + n, seq_len - 1)):
      mask_pos.add(i)
  return list(mask_pos)


### my code


def load_data(path):
  '''
  Returns
  data: 'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'
  label: 'User', 'Model', 'gt'
  '''
  data_path = os.path.join(path, "data_20_120.npy")
  label_path = os.path.join(path, "label_20_120.npy")
  data = np.load(
      data_path)  # 'Index', 'Arrival_Time', 'Creation_Time', 'x', 'y', 'z'
  label = np.load(label_path)  # 'User', 'Model', 'gt'
  return data, label


def split_data(data, label, train_rate=0.8, dev_rate=0.1):
  data = data.astype(np.float32)
  arr = np.arange(data.shape[0])
  np.random.shuffle(arr)
  data = data[arr]
  label = label[arr]
  train_index = int(data.shape[0] * train_rate)
  dev_index = int(data.shape[0] * (train_rate + dev_rate))
  train_data = data[:train_index, ...]
  dev_data = data[train_index:dev_index, ...]
  test_data = data[dev_index:, ...]
  
  label_index = -1
  train_label = label[:train_index, ..., label_index]
  dev_label = label[train_index:dev_index, ..., label_index]
  test_label = label[dev_index:, ..., label_index]
  return train_data, train_label, dev_data, dev_label, test_data, test_label


def unique_label(data, label):
  index = np.zeros(data.shape[0], dtype=bool)
  label_new = []
  for i in range(label.shape[0]):
    temp_label = np.unique(label[i])
    if temp_label.size == 1:
      index[i] = True
      label_new.append(label[i, 0])
  return data[index], np.array(label_new)


def finetune_data(data, label, train_rate=0.2):
  size = data.shape[0] * train_rate
  
  data, label = unique_label(data, label)
  if len(data) < size:
    size = len(data)
  
  index = []
  while len(index) < size:
    idx = np.random.randint(0, data.shape[0], dtype=int)
    if idx not in index:
      index.append(idx)

  train_data = data[index, ...]
  train_label = label[index, ...]
  return train_data, train_label


class BERTDataset4Pretrain(Dataset):
  """ Load sentence pair (sequential or random order) from corpus """

  def __init__(self, data):
    super().__init__()
    self.data = data

  def __getitem__(self, index):
    instance = self._normalize(self.data[index])
    mask_seq, masked_pos, seq = self._mask(instance)
    return torch.from_numpy(mask_seq), torch.from_numpy(
        masked_pos).long(), torch.from_numpy(seq)

  def __len__(self):
    return len(self.data)

  def _normalize(self, instance):
    instance_ = instance.copy()[:, :self.feature_len]
    instance_[:, :3] = instance_[:, :3] / 9.8  # acc
    return instance_

  def _mask(self,
            instance,
            mask_ratio=0.15,
            max_gram=10,
            mask_prob=0.8,
            replace_prob=0):
    shape = instance.shape

    # the number of prediction is sometimes less than max_pred when sequence is short
    n_pred = max(1, int(round(shape[0] * mask_ratio)))

    # For masked Language Models
    # mask_pos = bert_mask(shape[0], n_pred)
    mask_pos = span_mask(shape[0], max_gram, goal_num_predict=n_pred)

    instance_mask = instance.copy()

    if isinstance(mask_pos, tuple):
      mask_pos_index = mask_pos[0]
      if np.random.rand() < mask_prob:
        self.mask(instance_mask, mask_pos[0], mask_pos[1])
      elif np.random.rand() < replace_prob:
        self.replace(instance_mask, mask_pos[0], mask_pos[1])
    else:
      mask_pos_index = mask_pos
      if np.random.rand() < mask_prob:
        instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
      elif np.random.rand() < replace_prob:
        instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
    seq = instance[mask_pos_index, :]
    return instance_mask, np.array(mask_pos_index), np.array(seq)


class GPTDataset4Pretrain(Dataset):
  """ 
  # todo
  Dataset for Pre-training. E.g. for an IMU data length is 1:
  Input: torch.cat([-6.16290281, 0.3495758, 8.087262, 0.01107788, -0.0144043, -0.01611328]) -> 
  Output: torch.cat([-6.10410765e+00, 4.43362424e-01, 7.98728648e+00, 4.34875498e-03, -3.20281982e-02, -1.38702394e-02])
  Which will feed into the transformer concatenated as:
  input:  s1 s2 s3
  output: I  I  s3
  where I is "ignore", as the transformer is reading the input sequence
  """

  def __init__(self, data, length=6, feature_len=6):
    super().__init__()
    self.data = data
    self.length = length
    self.feature_len = feature_len

  def __getitem__(self, index):
    instance = self._normalize(self.data[index])
    return instance

  def get_vocab_size(self):
    return self.feature_len

  def get_block_size(self):
    return self.length * 2 - 1

  def __len__(self):
    return len(self.data)

  def _normalize(self, instance):
    instance_ = instance.copy()[:, :self.feature_len]
    instance_[:, :3] = instance_[:, :3] / 9.8  # acc
    return instance_

  def _flatten(self, instance):
    instance_ = instance.copy()
    shape = instance_.shape
    instance_.reshape(shape[0], -1)
    return instance_

  def _seq(self, instance):
    shape = instance.shape
    n_pred = max(1, int(shape[0] - self.length))


class Dataset4Pretrain(Dataset):

  def __init__(self, data) -> None:
    super().__init__()
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    instance = self._normalize(self.data[idx])
    instance = self._flatten(instance)
    instance = torch.from_numpy(instance).float()
    return instance, instance

  def _normalize(self, instance):
    instance_ = instance.copy()
    instance_[:, :3] = instance[:, :3] / 9.8  # acc
    return instance_
  
  def _flatten(self, instance):
    return instance.reshape(-1)


class IMUDataset(Dataset):

  def __init__(self, data, label):
    """
    Args:
        data (numpy.array): The IMU data of shape [num_samples, window_size, num_features]
    """
    self.data = data
    self.label = label

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    instance = self._normalize(self.data[idx])
    instance = self._flatten(instance)
    return torch.from_numpy(instance).float(), torch.from_numpy(
        np.array(self.label[idx])).long()

  def _normalize(self, instance):
    instance_ = instance.copy()
    instance_[:, :3] = instance[:, :3] / 9.8  # acc
    return instance_
  
  def _flatten(self, instance):
    return instance.reshape(-1)

def stat_acc_f1(label, label_estimated):
  # label = np.concatenate(label, 0)
  # results_estimated = np.concatenate(results_estimated, 0)
  f1 = f1_score(label, label_estimated, average='macro')
  acc = np.sum(label == label_estimated) / label.size
  return acc, f1