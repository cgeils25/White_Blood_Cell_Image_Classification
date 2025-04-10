import os
from typing import List, Tuple
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms.v2 as v2


DATA_DIR = "data"

CLASS_MAP = {
    'Basophil': 0,
    'Neutrophil': 1,
    'Monocyte': 2,
    'Eosinophil': 3,
    'Lymphocyte': 4
}

MODEL_TYPES = ["resnet18", "resnet34", "resnet50"]


class ImmuneCellImageDataset(Dataset):
    def __init__(self, img_paths: list, class_labels: torch.tensor, transform=None, device=torch.device('cpu')):
        self.img_paths=img_paths
        self.class_labels = class_labels
        self.transform=transform
        self.device = device
        self.unique_labels=torch.unique(class_labels)

        if len(img_paths) != class_labels.shape[0]:
            raise ValueError(f"Number of images ({len(img_paths)}) does not equal number of class labels ({class_labels.shape[0]})")
        
    def __getitem__(self, idx):
        # get image path with index
        img_path = self.img_paths[idx]

        # read image as tensor
        img_as_tensor = torchvision.io.read_image(img_path)
        assert self.transform is not None, "Error: must specify image transformation"
        img_as_tensor = self.transform(img_as_tensor)  # apply specified transformation to image
        img_as_tensor = img_as_tensor.to(self.device)  # move to given device
        
        # get image label
        img_label = self.class_labels[idx]
        img_label = img_label.to(self.device)

        return img_as_tensor, img_label

    def __len__(self):
        return len(self.img_paths)

    def get_class_counts(self):
        cell_type_counts = {cell_type.item(): 0 for cell_type in self.unique_labels}

        for img_label in self.class_labels:
            cell_type_counts[img_label.item()] += 1

        return cell_type_counts


def get_image_paths_and_class_labels(data_dir: str = DATA_DIR) -> Tuple[List[str], torch.Tensor]:
    class_subdirs = os.listdir(data_dir)

    image_paths = []
    class_labels = []

    for class_name in class_subdirs:
        if class_name not in CLASS_MAP:
            print(f"Unexpected class name: {class_name}. Skipping.")
            continue

        class_dir = os.path.join(data_dir, class_name)

        for filename in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, filename))
            class_labels.append(CLASS_MAP[class_name])
    
    class_labels = torch.tensor(class_labels)

    return image_paths, class_labels


def get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu:
        if torch.backends.mps.is_available():
            print("Using MPS.")
            device = torch.device('mps')

        elif torch.cuda.is_available():
            print("Using CUDA.")
            device = torch.device('cuda')

        else:
            print("No GPU available, using CPU.")
            device = torch.device('cpu')

    else:
        print("Using CPU.")
        device = torch.device('cpu')

    return device


def get_pretrained_model(model_type: str = "resnet18", num_classes: int = 5) -> nn.Module:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type {model_type} not recognized.")
    
    if model_type == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    return model


def split_data_stratified(image_paths: List[str], class_labels: torch.Tensor, train_split_percentage: float,
               valid_split_percentage: float, test_split_percentage: float, random_seed: int) -> \
                Tuple[List[str], torch.Tensor, List[str], torch.Tensor, List[str], torch.Tensor]:
    
    # split into train+valid and test
    train_valid_img_paths, test_img_paths, train_valid_class_labels, test_class_labels =\
    train_test_split(image_paths, class_labels, test_size=test_split_percentage, random_state=random_seed, stratify=class_labels)

    # account for the fact that the valid and train split percentages won't sum to 1
    adjusted_val_split_percentage = valid_split_percentage / train_split_percentage + valid_split_percentage

    # split train+valid subset into train and valid
    train_img_paths, valid_img_paths, train_class_labels, valid_class_labels =\
    train_test_split(train_valid_img_paths, train_valid_class_labels, test_size=adjusted_val_split_percentage, random_state=random_seed, stratify=train_valid_class_labels)

    return train_img_paths, train_class_labels, valid_img_paths, valid_class_labels, test_img_paths, test_class_labels


def get_transforms(noise_augmentation: bool = True, noise_mean: float = 0.0, noise_std: float = 0.1) -> Tuple[v2.Compose, v2.Compose, v2.Compose]:
    # transformations for training data
    train_transorm_list = [v2.Resize(size=(224, 224)),  # specs for imagenet
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)), # scale --> lower and upper bounds of crop size
        v2.ToDtype(torch.float32, scale=True), # convert to float tensor
        v2.Normalize(mean=[0.485, 0.456, 0.406],  # also specs for imagenet
        std=[0.229, 0.224, 0.225])
        ]
    
    if noise_augmentation:
        v2.GaussianNoise(mean=noise_mean, std=noise_std)

    train_transform = v2.Compose(transform=train_transorm_list)

    # sometimes RandomResizeCrop moves the cell out of frame, may cause a problem for some samples

    # transformations for test (and validation) data
    test_transform = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = test_transform
    
    return train_transform, val_transform, test_transform 


def get_dataloaders(image_paths: List[str], class_labels: torch.Tensor, train_split_percentage: float, valid_split_percentage: float, test_split_percentage: float,
                 train_transform: v2.Compose, valid_transform: v2.Compose, test_transform: v2.Compose, batch_size: int, num_workers: int, random_seed: int) \
                -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_img_paths, train_class_labels, valid_img_paths, valid_class_labels, test_img_paths, test_class_labels = \
        split_data_stratified(image_paths, class_labels,
                              train_split_percentage=train_split_percentage,
                              valid_split_percentage=valid_split_percentage,
                              test_split_percentage=test_split_percentage,
                              random_seed=random_seed)

    train_dataset = ImmuneCellImageDataset(
        img_paths=train_img_paths,
        class_labels=train_class_labels,
        transform=train_transform
    )

    valid_dataset = ImmuneCellImageDataset(
        img_paths=valid_img_paths,
        class_labels=valid_class_labels,
        transform=valid_transform
    )

    test_dataset = ImmuneCellImageDataset(
        img_paths=test_img_paths,
        class_labels=test_class_labels,
        transform=test_transform
    )

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    
    return train_dataloader, valid_dataloader, test_dataloader


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    # set model to train mode
    model.train()
    
    running_loss = 0.
    last_loss = 0.
    
    for batch_idx, batch in enumerate(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        batch_outputs = model(batch_inputs)

        # Compute the loss and its gradients
        loss = loss_fn(batch_outputs, batch_labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        num_batches_per_update = 50  # number of batches for each print statement
        
        if batch_idx % num_batches_per_update == 0 and not batch_idx == 0:
            last_loss = running_loss / num_batches_per_update # loss per batch
            print(f'Batch {batch_idx} loss: {last_loss}')
            running_loss = 0.

    return last_loss


def validate_classification_model(dataloader, model, loss_fn, device):
    # set model to evaluation mode
    model.eval()
    
    batch_loss = 0
    batch_loss_list = []

    unique_labels = dataloader.dataset.unique_labels
    losses_for_unique_labels_raw = {unique_label.item(): [] for unique_label in unique_labels}
    
    for batch in tqdm(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # Make predictions for this batch
        batch_outputs = model(batch_inputs).detach()
        
        # Compute the loss
        loss = loss_fn(batch_outputs, batch_labels)

        # get loss for each label
        for unique_label in unique_labels:
            if any(batch_labels == unique_label):
                batch_idxs_for_unique_label = batch_labels == unique_label
                
                batch_outputs_for_unique_label = batch_outputs[batch_idxs_for_unique_label]
                num_samples_in_batch = len(batch_outputs_for_unique_label)
                batch_targets_for_unique_label = unique_label.repeat(1, num_samples_in_batch).flatten().to(device)
               
                loss_for_unique_label = loss_fn(batch_outputs_for_unique_label, batch_targets_for_unique_label)
                loss_for_unique_label = loss_for_unique_label.item()
            else:
                loss_for_unique_label = 0
                num_samples_in_batch = 0
                
            losses_for_unique_labels_raw[unique_label.item()].append((loss_for_unique_label, num_samples_in_batch))
        
        # Extract batch loss as float
        batch_loss = loss.item()

        # add batch loss to running list
        batch_loss_list.append(batch_loss) 

    losses_for_unique_labels = {unique_label.item(): [] for unique_label in unique_labels}

    # obtain losses for each class by aggregating individual batch losses
    for unique_label in unique_labels:
        loss_list_for_unique_label_raw = losses_for_unique_labels_raw[unique_label.item()]
        total_samples = sum([pair[1] for pair in loss_list_for_unique_label_raw])
        loss_weights = [pair[1]/total_samples for pair in loss_list_for_unique_label_raw]
        assert sum(loss_weights) == 1 
        weighted_losses = [pair[0] * loss_weights[i] for i, pair in enumerate(loss_list_for_unique_label_raw)]
        aggregated_class_loss = sum(weighted_losses)
        losses_for_unique_labels[unique_label.item()] = aggregated_class_loss

    mean_validation_loss = np.mean(batch_loss_list)
    return mean_validation_loss, losses_for_unique_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train model to classify white blood cell images")
    return parser.parse_args()

def main():
    # put as little in main as possible. want to be able to use optuna
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
