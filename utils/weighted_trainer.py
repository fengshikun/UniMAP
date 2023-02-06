from transformers import Trainer
import torch
import numpy as np


def get_sampler_weight(train_dataset, val_dataset):
    train_labels = train_dataset.labels
    val_labels = val_dataset.labels
    all_labels = np.concatenate([train_labels, val_labels], axis=0)
    sample_num, task_num = all_labels.shape
    # label one ratio for each task
    label_one_ratio = all_labels.sum(axis=0) / sample_num
    label_zero_ratio = 1 - label_one_ratio
    label_one_ratio = label_one_ratio.reshape(-1, 1)
    label_zero_ratio = label_zero_ratio.reshape(-1, 1)
    label_ratio = np.concatenate([label_zero_ratio, label_one_ratio], axis=1)
    # assume we have binary label of training dataset and val dataset; use label as index
    all_label_int = all_labels.astype(np.int)
    
    all_label_ratio = np.zeros_like(all_labels)
    for i in range(task_num):
        all_label_ratio[:, i] = label_ratio[i][all_label_int[:, i]]
        
    # split train label ratio
    train_num = train_labels.shape[0]
    train_label_ratio = all_label_ratio[:train_num]
    
    # change the ratio to the weight
    train_label_weight = (1.0 / train_label_ratio).sum(axis=1)
    
    # normalise
    normalise_alpha = train_num / train_label_weight.sum()
    
    train_label_weight_norm = normalise_alpha * train_label_weight
    
    return train_label_weight_norm

class WeightedTrainer(Trainer):
    def set_weight(self, weight):
        self.weight = weight
        self.sample_len = weight.shape[0]
    def _get_train_sampler(self):
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        sampler = torch.utils.data.WeightedRandomSampler(self.weight, self.sample_len, replacement=True, generator=generator)
        return sampler
        