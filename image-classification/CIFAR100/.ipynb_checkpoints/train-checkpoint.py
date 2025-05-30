import random
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from ghf import ActGHF
## Import model
from model import create_trainer, Classifier, CNN, ResidualBlock, ResNet
from save_plot import save_tensorboard_data

# SEED = 42
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Image preprocessing modules
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# test dataset
test_dataset = torchvision.datasets.CIFAR100(root='data/',train=False, transform=test_transform, download=True)
# create dataLoader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64, shuffle=False, num_workers=2)


if __name__=='__main__':

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        normalize])

    train_dataset = torchvision.datasets.CIFAR100(root='data/',train=True,download=True, transform=None)
    # split into validation and training dataset
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # add transformation
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    # train and val loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=64, shuffle=False, num_workers=2)

    max_epochs = 100
    num_classes = 100
    list_model = {'GHF':ActGHF(), 'Logistic':nn.Sigmoid(), 'Tanh':nn.Tanh(), 'ReLU':nn.ReLU(),'Mish': nn.Mish(),'LeakyReLU':nn.LeakyReLU()}

    for name, act in list_model.items():
        # model = CNN(activation_fn=act, num_classes=num_classes)
        model = ResNet(ResidualBlock, [3, 3, 3], num_classes=num_classes, activation_fn=act)
        trainer = create_trainer(name, max_epochs=max_epochs)
        print(f'Train with {name}')
        classifier = Classifier(model, num_classes=num_classes)
        trainer.fit(classifier, train_loader, val_loader)
        trainer.test(classifier, dataloaders=test_loader)

    # save plot
    save_tensorboard_data(list(list_model.keys()))