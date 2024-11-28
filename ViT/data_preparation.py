import os
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def prepare_data_loaders(root_dir, batch_size = 128, num_workers = 12, shuffle = True, validation_split = 0.2):
    """ 
    Prepare data loaders for training, validation, and testing.

    Args:
        root_dir: The root directory of the dataset.
        batch_size: The number of samples per batch. Default is 128.
        num_workers: The number of subprocesses to use for data loading. Default is 12.
        shuffle: Whether to shuffle the data. Default is True.
        validation_split: The proportion of the dataset to use for validation. Default is 0.2.

    Returns:
        tuple: Train, validation, and test data loaders.
    """
    # Create transformations for training and testing
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root_dir)

    # Split into training, validation, and test datasets
    data_size = len(dataset)
    train_size = floor(5/10  * data_size)
    val_size = floor(2/10 * data_size)
    test_size = data_size - train_size - val_size

    assert train_size + val_size + test_size == data_size, "Sizes do not match the dataset size."

    # Split the dataset into training, validation, and test datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = transform
    test_dataset.dataset.transform = transform

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return ( trainloader, valloader, testloader, dataset.classes )


def prepare_test_data_loader(root_dir, batch_size = 4, num_workers = 2):
    """ 
    Prepare a data loader for the test dataset.

    Args:
        root_dir (str): The root directory of the test dataset.
        batch_size (int): The number of samples per batch. Default is 4.
        num_workers (int): The number of subprocesses to use for data loading. Default is 2.

    Returns:
        tuple: Test data loader and a list of class names.
    """
    # Create transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((224, 224), antialias=True, scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=cv2.INTER_CUBIC),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the images dataset
    images = ImageFolder(root_dir, transform)

    # Create DataLoader for testing set
    test_loader = DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return ( testloader, images.classes )