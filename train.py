"""
A CLI-based pipeline to train a model in the ResNet family to classify Leukocytes.

For more information, run:

uv run python train.py --help
"""

import os
from typing import List, Tuple
import argparse
from tqdm import tqdm
from neattime import neattime
import time
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms.v2 as v2


CLASS_MAP = {
    'Basophil': 0,
    'Neutrophil': 1,
    'Monocyte': 2,
    'Eosinophil': 3,
    'Lymphocyte': 4
}

CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}

MODEL_TYPES = ["resnet18", "resnet34", "resnet50"] 

OPTIMIZER_TYPES = ["Adam", "SGD"]   

NUM_SAMPLES_FOR_TEST_RUN = 1000

NUM_EPOCHS_FOR_TEST_RUN = 2


class ImmuneCellImageDataset(Dataset):
    def __init__(self, img_paths: List[str], class_labels: torch.tensor, transform: v2.Compose = None, unique_labels: torch.tensor = None, device: torch.device = torch.device('cpu')):
        """Initialize an image dataset

        Args:
            img_paths (List[str]): list of image file paths 
            class_labels (torch.tensor): class labels corresponding to the images, in the same order as img_paths
            transform (v2.Compose, optional): transformations to be applied to the images. Defaults to None.
            unique_labels (torch.tensor, optional): unique labels in the dataset. If None, will take the unique values in class_labels Defaults to None.
            device (torch.device, optional): the device onto which tensors will be moved. Defaults to torch.device('cpu').

        Raises:
            ValueError: if the number of images does not equal the number of class labels
        """
        self.img_paths=img_paths
        self.class_labels = class_labels
        self.transform=transform
        self.device = device
        self.unique_labels=torch.unique(class_labels) if unique_labels is None else unique_labels

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


def get_image_paths_and_class_labels(data_dir: str) -> Tuple[List[str], torch.Tensor]:
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
    """Retrieve the device to use for model training and inference

    Args:
        use_gpu (bool, optional): Whether to look for a GPU (MPS or CUDA). Defaults to True.

    Returns:
        torch.device: the device to use
    """
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


def get_pretrained_model(model_type: str, num_classes: int, device: torch.device) -> nn.Module:
    f"""Instantiate a pretrained pytorch model based on the specified type

    Args:
        model_type (str): the type of model. Supported values are {MODEL_TYPES}
        num_classes (int): the number of classes in the dataset aka the number of outputs
        device (torch.device): the device onto which the model will be moved

    Raises:
        ValueError: if the model type is not recognized

    Returns:
        nn.Module: the pytorch model
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type {model_type} not recognized. Choose from {MODEL_TYPES}.")
    
    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # making sure gradients flow through the entire model
    for parameter in model.parameters():
        parameter.requires_grad = True
        assert parameter.requires_grad == True
    
    model = model.to(device)
    
    return model


def split_data_stratified(image_paths: List[str], class_labels: torch.Tensor, train_split_percentage: float,
               valid_split_percentage: float, test_split_percentage: float, random_seed: int) -> Tuple[List[str], torch.Tensor, List[str], torch.Tensor, List[str], torch.Tensor]:
    """Split dataset into train, validation, and test sets, stratified by class labels

    Args:
        image_paths (List[str]): list of image file paths
        class_labels (torch.Tensor): class labels corresponding to the images, in the same order as image_paths
        train_split_percentage (float): percentage of data to use for training
        valid_split_percentage (float): percentage of data to use for validation
        test_split_percentage (float): percentage of data to use for testing
        random_seed (int): random seed for reproducibility

    Raises:
        ValueError: if the split percentages do not sum to 1.0

    Returns:
        Tuple[List[str], torch.Tensor, List[str], torch.Tensor, List[str], torch.Tensor]: paths to train images, train labels, paths to validation images, validation labels, paths to test images, test labels
    """
    if not np.isclose(train_split_percentage + valid_split_percentage + test_split_percentage, 1.0):
        # had to use isclose due to numerical imprecision problems
        raise ValueError("Split percentages must sum to 1.0")

    # split into train+valid and test
    train_valid_img_paths, test_img_paths, train_valid_class_labels, test_class_labels =\
    train_test_split(image_paths, class_labels, test_size=test_split_percentage, random_state=random_seed, stratify=class_labels)

    # account for the fact that the valid and train split percentages won't sum to 1
    adjusted_valid_split_percentage = valid_split_percentage / train_split_percentage + valid_split_percentage

    # split train+valid subset into train and valid
    train_img_paths, valid_img_paths, train_class_labels, valid_class_labels =\
    train_test_split(train_valid_img_paths, train_valid_class_labels, test_size=adjusted_valid_split_percentage, random_state=random_seed, stratify=train_valid_class_labels)

    return train_img_paths, train_class_labels, valid_img_paths, valid_class_labels, test_img_paths, test_class_labels


def get_train_transform(noise_augmentation: bool, noise_mean: float, noise_std: float) -> Tuple[v2.Compose, v2.Compose, v2.Compose]:
    # transformations for training data. Separated for convenience
    train_transorm_list = [v2.Resize(size=(224, 224)),  # specs for imagenet
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0)), # scale --> lower and upper bounds of crop size
        v2.ToDtype(torch.float32, scale=True), # convert to float tensor
        v2.Normalize(mean=[0.485, 0.456, 0.406],  # also specs for imagenet
        std=[0.229, 0.224, 0.225])
        ] # sometimes RandomResizeCrop moves the cell out of frame, may cause a problem for some samples
    
    if noise_augmentation:
        print("Adding Gaussian noise augmentation to training data.")
        train_transorm_list.append(v2.GaussianNoise(mean=noise_mean, sigma=noise_std))
    else:
        print("No Gaussian noise augmentation applied to training data.")

    train_transform = v2.Compose(transforms=train_transorm_list)

    return train_transform


def get_test_transform() -> v2.Compose:
    # transformations for test (and validation) data
    test_transform = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])

    return test_transform


def get_transforms(noise_augmentation: bool, noise_mean: float, noise_std: float) -> Tuple[v2.Compose, v2.Compose, v2.Compose]:
    """Get the image transformations to use for training, validation, and test datasets

    Args:
        noise_augmentation (bool): whether to apply gaussian noise to training images
        noise_mean (float): mean of Gaussian noise. Ignored if noise_augmentation is False
        noise_std (float): standard deviation of Gaussian noise. Ignored if noise_augmentation is False

    Returns:
        Tuple[v2.Compose, v2.Compose, v2.Compose]: training data transformations, validation data transformations, test data transformations
    """

    train_transform = get_train_transform(noise_augmentation=noise_augmentation, 
        noise_mean=noise_mean,
        noise_std=noise_std)

    test_transform = get_test_transform()
    
    valid_transform = test_transform
    
    return train_transform, valid_transform, test_transform 


def get_dataloaders(image_paths: List[str], class_labels: torch.Tensor, train_split_percentage: float, valid_split_percentage: float, test_split_percentage: float,
                 train_transform: v2.Compose, valid_transform: v2.Compose, test_transform: v2.Compose, batch_size: int, num_workers: int, device: torch.device, random_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get the dataloaders for training, validation, and test datasets

    Args:
        image_paths (List[str]): list of image file paths
        class_labels (torch.Tensor): class labels corresponding to the images, in the same order as image_paths
        train_split_percentage (float): percentage of data to use for training
        valid_split_percentage (float): percentage of data to use for validation
        test_split_percentage (float): percentage of data to use for testing
        train_transform (v2.Compose): transformations for training data
        valid_transform (v2.Compose): transformations for validation data
        test_transform (v2.Compose): transformations for test data
        batch_size (int): batch size for dataloaders
        num_workers (int): number of cpu cores to use for dataloaders
        device (torch.device): the device onto which tensors will be moved
        random_seed (int): random seed for reproducibility

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train dataloader, validation dataloader, test dataloader
    """
    train_img_paths, train_class_labels, valid_img_paths, valid_class_labels, test_img_paths, test_class_labels = \
        split_data_stratified(image_paths, class_labels,
                              train_split_percentage=train_split_percentage,
                              valid_split_percentage=valid_split_percentage,
                              test_split_percentage=test_split_percentage,
                              random_seed=random_seed)
    
    # this is a little hacky. Just fixes a n_problem during test runs where not all 5 classes end up in the reduced dataset
    unique_labels = torch.tensor(list(CLASS_MAP.values())).sort()[0]

    train_dataset = ImmuneCellImageDataset(
        img_paths=train_img_paths,
        class_labels=train_class_labels,
        transform=train_transform,
        unique_labels=unique_labels,
        device=device
    )

    valid_dataset = ImmuneCellImageDataset(
        img_paths=valid_img_paths,
        class_labels=valid_class_labels,
        transform=valid_transform,
        unique_labels=unique_labels,
        device=device
    )

    test_dataset = ImmuneCellImageDataset(
        img_paths=test_img_paths,
        class_labels=test_class_labels,
        transform=test_transform,
        unique_labels=unique_labels,
        device=device
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


def get_loss_function(weight: torch.tensor = None) -> nn.Module:
    """Get the loss function to use for training, optionally with class weights

    Args:
        weight (torch.tensor, optional): weights to apply to each class when computing loss. Defaults to None.

    Returns:
        nn.Module: the loss function
    """
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    return loss_fn


def compute_class_weights(dataloader: DataLoader) -> torch.tensor:
    """Compute class weights based on the class counts in the dataset. Class weights are computed as the total number of samples divided by the number of samples for each class.

    Args:
        dataloader (DataLoader): the dataloader for the dataset

    Returns:
        torch.tensor: tensor of class weights
    """
    class_counts = dataloader.dataset.get_class_counts()
    total_samples = sum(class_counts.values())
    
    # prevent division by 0 if for some reason one of the classes isn't present
    class_weights = {label: (total_samples / count) if count > 0 else 0 for label, count in class_counts.items()}

    print(class_weights)
    
    weights_tensor = torch.tensor([class_weights[label.item()] for label in dataloader.dataset.unique_labels], dtype=torch.float32)
    
    return weights_tensor


def get_optimizer(type: str, model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    f"""Get the optimizer to use for training

    Args:
        type (str): type of optimizer. Supported values are {OPTIMIZER_TYPES}
        model (nn.Module): the pytorch model to optimize
        learning_rate (float): the learning rate for the optimizer
        weight_decay (float): l2 penalty for the optimizer

    Raises:
        ValueError: if the optimizer type is not recognized

    Returns:
        torch.optim.Optimizer: the optimizer
    """
    if type not in OPTIMIZER_TYPES:
        raise ValueError(f"Optimizer type {type} not recognized. Choose from {OPTIMIZER_TYPES}.")

    if type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    elif type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    return optimizer


def train_one_epoch_classification_model(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """Train the classification model for one epoch

    Args:
        dataloader (DataLoader): the dataloader for the training dataset
        model (nn.Module): the pytorch model to train
        loss_fn (nn.Module): the loss function to minimize
        optimizer (torch.optim.Optimizer): the optimizer to use
        device (torch.device): the device onto which tensors will be moved

    Returns:
        float: the last loss value for the epoch
    """
    # set model to train mode
    model.train()
    
    running_loss = 0.
    last_loss = 0.
    
    for batch_idx, batch in enumerate(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # this should be handled by the dataloader but moving to mps is breaking for me
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

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


def validate_classification_model(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device, return_predictions_and_labels: bool = False):
    """Validate the classification model on the given dataloader

    Args:
        dataloader (DataLoader): the dataloader for the validation or test dataset
        model (nn.Module): the pytorch model to validate
        loss_fn (nn.Module): the loss function to use
        device (torch.device): the device onto which tensors will be moved
        return_predictions_and_labels (bool, optional): Whether to return all predictions and class labels. Defaults to False.

    Returns:
        Tuple[float, dict]: mean validation loss, dictionary of losses for each class
        or
        Tuple[float, dict, dict, List[int]]: mean validation loss, dictionary of losses for each class, dictionary of predictions for each class, list of true class labels
    """
    
    # set model to evaluation mode
    model.eval()
    
    batch_loss = 0
    batch_loss_list = []

    unique_labels = dataloader.dataset.unique_labels
    losses_for_unique_labels_raw = {unique_label.item(): [] for unique_label in unique_labels}

    if return_predictions_and_labels:
        # so we can collect the predictions and labels
        all_predictions = {unique_label.item(): [] for unique_label in unique_labels}
        all_labels = []
    
    for batch in tqdm(iter(dataloader)):
        # Every data instance is an input + label pair
        batch_inputs, batch_labels = batch

        # this should be handled by the dataloader but moving to mps is breaking for me
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Make predictions for this batch
        batch_outputs = model(batch_inputs).detach()
        
        # Compute the loss
        loss = loss_fn(batch_outputs, batch_labels)

        # save the predictions and labels if requested
        if return_predictions_and_labels:
            batch_output_probabilities = torch.softmax(batch_outputs, dim=1)    

            for label, predictions in enumerate(batch_output_probabilities.T.tolist()): # this assumes the labels are [0, 1, 2, ...]. not sure why that would ever be violated but just making a note of it
                all_predictions[label].extend(predictions)

            all_labels.extend(batch_labels.tolist())

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
        loss_weights = [(pair[1]/total_samples) if total_samples > 0 else 0 for pair in loss_list_for_unique_label_raw]
        if not sum(loss_weights) == 1:
            warnings.warn(f"Warning: Loss weights for class {unique_label.item()} do not sum to 1.0. This is likely caused by one class not being present in the batch.")
        weighted_losses = [pair[0] * loss_weights[i] for i, pair in enumerate(loss_list_for_unique_label_raw)]
        aggregated_class_loss = sum(weighted_losses)
        losses_for_unique_labels[unique_label.item()] = aggregated_class_loss

    mean_validation_loss = np.mean(batch_loss_list)

    if return_predictions_and_labels:
        return mean_validation_loss, losses_for_unique_labels, all_predictions, all_labels

    return mean_validation_loss, losses_for_unique_labels


def create_and_save_loss_df(all_train_loss: List[float], all_validation_loss: List[float], test_loss: float, all_class_validation_losses: dict, num_epochs:int, output_dir: str):
    """Create and save a dataframe of training, validation, and test losses

    Args:
        all_train_loss (List[float]): all training loss values
        all_validation_loss (List[float]): all validation loss values
        test_loss (float): final test loss value 
        all_class_validation_losses (dict): dictionary of validation losses for each class
        num_epochs (int): number of epochs trained
        output_dir (str): directory to save the loss dataframe
    """
    
    df_train_val_loss = pd.DataFrame({
    'Epoch': np.arange(1, num_epochs+1),
    'Train': all_train_loss,
    'Validation': all_validation_loss,
    'Test': [None] * num_epochs
    })

    df_train_val_loss.loc[num_epochs-1, 'Test'] = test_loss

    # want mapping of integer target labels to class names
    CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}

    df_classes = pd.DataFrame({CLASS_MAP_INV[label]: all_class_validation_losses[label] for label in all_class_validation_losses if len(all_class_validation_losses[label]) == num_epochs})

    all_loss_df = pd.concat([df_train_val_loss, df_classes], axis=1)

    loss_df_path = os.path.join(output_dir, 'losses.csv')
    
    all_loss_df.to_csv(loss_df_path, index=False)

    print(f"Loss data saved to {loss_df_path}")

    
def train_classification_model(
    data_dir: str,
    use_gpu: bool,
    noise_augmentation: bool,
    noise_mean: float,
    noise_std: float,
    train_split_percentage: float,
    valid_split_percentage: float,
    test_split_percentage: float,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    model_type: str,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    weight_loss: bool,
    output_dir: str,
    save_test_predictions_and_labels: bool,
    test_run: bool
):
    """Train a classification model to classify white blood cell images

    Args:
        data_dir (str): Directory containing the images. Structure of data directory should be as follows: <<data_dir>>/<<class_name>>/<<image_file.png>>
        use_gpu (bool): Whether to attempt to use GPU for training
        noise_augmentation (bool): Whether to apply Gaussian noise augmentation to training images
        noise_mean (float): Mean of Gaussian noise for training images
        noise_std (float): Standard deviation of Gaussian noise for training images
        train_split_percentage (float): Percentage of data to use for training. Split percentages must sum to 1.0
        valid_split_percentage (float): Percentage of data to use for validation. Split percentages must sum to 1.0
        test_split_percentage (float): Percentage of data to use for testing. Split percentages must sum to 1.0
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of workers for data loading
        random_seed (int): Random seed for reproducibility. Note: PyTorch cannot guarantee reproducibility (even on the same hardware)
        model_type (str): Type of model to use
        optimizer_type (str): Type of optimizer to use
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): L2 penalty for optimizer
        num_epochs (int): Number of epochs to train the model
        weight_loss (bool): Whether to weight the loss function by class counts in the training dataset. Useful for unbalanced datasets
        output_dir (str): Directory to save results and models
        save_test_predictions_and_labels (bool): Whether to save test predictions to the output directory
        test_run (bool): Whether to run a test run with reduced data
    """
    start = time.monotonic()
    print("="*100)
    print("Starting training process...")
    print("="*100)

    # get dataloaders
    print("Preparing data...")
    image_paths, class_labels = get_image_paths_and_class_labels(data_dir=data_dir)

    if test_run:
        print("Running test run with reduced data...")
        image_paths = image_paths[:NUM_SAMPLES_FOR_TEST_RUN]
        class_labels = class_labels[:NUM_SAMPLES_FOR_TEST_RUN]
        num_epochs = NUM_EPOCHS_FOR_TEST_RUN
        output_dir = "TEST_RUN_" + output_dir 
    
    # make output directory
    os.makedirs(output_dir, exist_ok=True)

    device = get_device(use_gpu=use_gpu)

    train_transform, valid_transform, test_transform = get_transforms(noise_augmentation=noise_augmentation,
        noise_mean=noise_mean,
        noise_std=noise_std)
    
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
        image_paths=image_paths,
        class_labels=class_labels,
        train_split_percentage=train_split_percentage,
        valid_split_percentage=valid_split_percentage,
        test_split_percentage=test_split_percentage,
        train_transform=train_transform,
        valid_transform=valid_transform,
        test_transform=test_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=torch.device('cpu'), # these dataloaders break when using MPS: https://stackoverflow.com/questions/76671692/iterate-over-dataloader-which-is-loaded-on-gpu-mps 
        random_seed=random_seed
    )
    print("Data prepared.")

    print("."*100)
    print("Dataloader Class Counts and Proportions:")

    for dataloader, name in zip([train_dataloader, validation_dataloader, test_dataloader],
                                ['Train', 'Validation', 'Test']):
        class_counts = dataloader.dataset.get_class_counts()
        total_samples = sum(class_counts.values())
        print(f"\t{name} dataset class counts:")
        for label, count in class_counts.items():
            proportion = count / total_samples
            print(f"\t\tClass {CLASS_MAP_INV[label]}: {count} samples ({round(proportion*100, 2)}%)")
        print()

    print("."*100)

    # get model
    print(f"Instantiating {model_type} model...")
    model = get_pretrained_model(model_type=model_type, num_classes=len(CLASS_MAP), device=device)
    print(f"{model_type} model instantiated.")
    print("."*100)

    # get loss function and optimizer
    print("Preparing loss function and optimizer...")

    if weight_loss:
        print("Weighting loss function by class counts in training dataset.")
        class_weights = compute_class_weights(train_dataloader)
        loss_fn = get_loss_function(weight=class_weights.to(device))

        print("Class weights:")
        for i, w in enumerate(class_weights):
            print(f"\tClass {CLASS_MAP_INV[i]}: weight = {w.item()}")

    else:
        loss_fn = get_loss_function()

    optimizer = get_optimizer(optimizer_type, model, learning_rate=learning_rate, weight_decay=weight_decay)
    print("Loss function and optimizer prepared.")

    print("."*100)

    all_train_loss = []
    all_validation_loss = []
    all_class_validation_losses = {label: [] for label in list(CLASS_MAP.values())}

    best_model_path = None
    best_model_valid_loss = None
    best_model_epoch = None

    for epoch in range(1, num_epochs+1):
        print("-"*100)
        print("Epoch ", epoch)

        # train
        print("Starting Model Training...")
        last_train_loss = train_one_epoch_classification_model(train_dataloader, model, loss_fn, optimizer, device)
        print("Last train loss: ", last_train_loss)

        all_train_loss.append(last_train_loss)

        # validation
        print("Starting Model Validation...")
        mean_validation_loss, losses_for_unique_labels = validate_classification_model(validation_dataloader, model, loss_fn, device)
        print("Mean validation loss: ", mean_validation_loss)

        print("Losses for individual classes: ")

        for label, loss_val in losses_for_unique_labels.items():
            print(f"\t{label}: {loss_val}")

        all_validation_loss.append(mean_validation_loss)

        # record individual class labels
        for label in losses_for_unique_labels:
            all_class_validation_losses[label].append(losses_for_unique_labels[label]) # this is stupid, I know

        # saving model
        print("Attempting to save model...")
        if best_model_valid_loss is None and best_model_path is None:
            print("Saving initial model...")
            model_name = f'{model_type}_epoch_{epoch}.pt'
            model_path = os.path.join(output_dir, model_name)
            torch.save(model, model_path) # not saving state_dict because the C++ frontend doesn't like it

            best_model_path = model_path
            best_model_valid_loss = mean_validation_loss
            best_model_epoch = epoch
        
        else:
            if mean_validation_loss < best_model_valid_loss:
                print(f"Saving new best model for epoch {epoch}...")
                model_name = f'{model_type}_epoch_{epoch}.pt'
                model_path = os.path.join(output_dir, model_name)
                torch.save(model, model_path)

                assert os.path.exists(model_path), 'Model failed to save'
                print("Model saved successfully.")

                print("Removing previous best model...")
                if best_model_path is not None: # this check shouldn't be necessary but trying to be safe
                    os.remove(best_model_path)
                    print("Previous best model removed.")

                print("Updating best model info...")
                best_model_path = model_path
                best_model_valid_loss = mean_validation_loss
                best_model_epoch = epoch
                print("Done.")

            else:
                print("Not saving model, validation loss did not improve.")
        
        print(f"""Current best model: 
              \tModel path: {best_model_path}
              \tEpoch: {best_model_epoch}
              \tValidation loss: {best_model_valid_loss}
                        """)

        assert os.path.exists(model_path), 'Model failed to save'
        print("Model saved")

    print("Training complete.")

    # create and save loss dataframe
    print("Creating and saving loss dataframe...")
    create_and_save_loss_df(
        all_train_loss=all_train_loss,
        all_validation_loss=all_validation_loss,
        test_loss=mean_validation_loss,
        all_class_validation_losses=all_class_validation_losses,
        num_epochs=num_epochs,
        output_dir=output_dir
    )
    print("Loss dataframe saved.")

    # test
    print("Starting Model Test...")
    if save_test_predictions_and_labels:
        print("Saving test predictions and labels to output directory.")
        test_loss, losses_for_unique_labels, all_predictions, all_labels = \
        validate_classification_model(test_dataloader, model, loss_fn, device, return_predictions_and_labels=True)

        test_prediction_df = pd.DataFrame({'Labels': all_labels})

        for label in all_predictions:
            test_prediction_df[CLASS_MAP_INV[label]] = all_predictions[label]

        test_prediction_path = os.path.join(output_dir, 'test_predictions.csv')

        test_prediction_df.to_csv(test_prediction_path, index=False)

        print(f"Test predictions saved to {test_prediction_path}")
    
    else:
        test_loss, losses_for_unique_labels = validate_classification_model(test_dataloader, model, loss_fn, device, return_predictions_and_labels=False)

    print("Test loss: ", test_loss)
    print(f"Losses for individual classes: {losses_for_unique_labels}")

    print("="*60)
    print("Model training and evaluation complete.")
    print(f"Total training time: {time.monotonic() - start}")
    print("="*60)


def print_args(args: argparse.Namespace) -> None:
    """
    Print the arguments passed to the script
    """
    print('-'*100)
    print(f'Arguments passed to {__file__}:')
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
    print('-'*100)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model to classify white blood cell images")

    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the images. Structure of data directory should be as follows: <<data_dir>>/<<class_name>>/<<image_file.png>>")
    parser.add_argument("--use_gpu", action='store_true', help="Whether to attempt to use GPU for training")
    parser.add_argument("--noise_augmentation", action='store_true', help="Whether to apply Gaussian noise augmentation to training images")
    parser.add_argument("--noise_mean", type=float, default=0.0, help="Mean of Gaussian noise for training images")
    parser.add_argument("--noise_std", type=float, default=1e-7, help="Standard deviation of Gaussian noise for training images")
    parser.add_argument("--train_split_percentage", type=float, default=0.75, help="Percentage of data to use for training. Split percentages must sum to 1.0")
    parser.add_argument("--valid_split_percentage", type=float, default=0.15, help="Percentage of data to use for validation. Split percentages must sum to 1.0")
    parser.add_argument("--test_split_percentage", type=float, default=0.1, help="Percentage of data to use for testing. Split percentages must sum to 1.0")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of workers for data loading")
    parser.add_argument("--random_seed", type=int, default=1738, help="Random seed for reproducibility. Note: pytorch cannot reproducibility (even on the same hardware).")
    parser.add_argument("--model_type", type=str, default="resnet18", choices=MODEL_TYPES, help="Type of model to use")
    parser.add_argument("--optimizer_type", type=str, default="Adam", choices=OPTIMIZER_TYPES, help="Type of optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 penalty for optimizer")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train the model")
    parser.add_argument("--weight_loss", action='store_true', help="Whether to weight the loss function by class counts in the training dataset. Useful for unbalanced datasets. The weight will be equal to the total number of samples divided by the number of samples for each class.")
    parser.add_argument("--output_dir", type=str, default=f"model_result_{neattime()}", help="Directory to save results and models")
    parser.add_argument("--save_test_predictions_and_labels", action='store_true', help="Whether to save test predictions to output directory")
    parser.add_argument("--test_run", action='store_true', help="Whether to run a test run with reduced data")

    return parser.parse_args()


def main(args):
    print_args(args)

    train_classification_model(
        data_dir=args.data_dir,
        use_gpu=args.use_gpu,
        noise_augmentation=args.noise_augmentation,
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
        train_split_percentage=args.train_split_percentage,
        valid_split_percentage=args.valid_split_percentage,
        test_split_percentage=args.test_split_percentage,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        model_type=args.model_type,
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        weight_loss=args.weight_loss,
        output_dir=args.output_dir,
        save_test_predictions_and_labels=args.save_test_predictions_and_labels,
        test_run = args.test_run
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
