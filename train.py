import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from collections import OrderedDict

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", action="store", default="./flowers/")
parser.add_argument(
    "--save_dir",
    type=str,
    default="save_directory",
    help="Directory to save checkpoint",
)
parser.add_argument(
    "--arch",
    dest="arch",
    default="vgg16",
    choices=["vgg16", "densenet121"],
    help="enter arch model",
)
parser.add_argument(
    "--learning_rate", type=float, default=0.001, help="enter learning rate"
)
parser.add_argument(
    "--hidden_units", type=int, default=512, help="enter # of hidden units"
)
parser.add_argument("--epochs", type=int, default=10, help="enter number of epochs")
parser.add_argument("--gpu_cpu", type=str, default="cuda", help="specify cuda or cpu")
args = parser.parse_args()

learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu_cpu = args.gpu_cpu
arch = args.arch
sav_dir = args.save_dir

data_dir = "flowers"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
validation_transforms = transforms.Compose(
    [
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
validation_data = datasets.ImageFolder(
    data_dir + "/valid", transform=validation_transforms
)
test_data = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

image_datasets = {
    "train": datasets.ImageFolder(data_dir + "/train", transform=train_transforms),
    "validation": datasets.ImageFolder(
        data_dir + "/valid", transform=validation_transforms
    ),
    "test": datasets.ImageFolder(data_dir + "/test", transform=test_transforms),
}

import json

with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = getattr(models, args.arch)(pretrained=True)
model

for param in model.parameters():
    param.requires_grad = False

if args.arch == "vgg16":
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(25088, 1000)),
                ("drop", nn.Dropout(p=0.2)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(1000, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
elif args.arch == "densenet121":
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(1024, 500)),
                ("drop", nn.Dropout(p=0.6)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(500, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

model.classifier = classifier
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

print("Initializing Training...")

steps = 0
running_loss = 0
print_every = 40

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
            validation_loss = 0
            accuracy = 0
            with torch.no_grad():
                for images, labels in validationloader:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()

                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor))

            print(
                f"Epoch {epoch+1}/{epochs}.."
                f"Train loss: {running_loss/print_every:.3f}.."
                f"Validation loss: {validation_loss/len(validationloader): .3f}.."
                f"Validation accuracy: {accuracy/len(validationloader):.3f}"
            )

            running_loss = 0
            model.train()


model.eval()
validation_loss = 0
accuracy = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        logps = model(images)
        loss = criterion(logps, labels)
        validation_loss += loss.item()

        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor))

print(
    f"Validation loss: {validation_loss/len(testloader): .3f}.."
    f"Validation accuracy: {accuracy/len(testloader):.3f}"
)

model.class_to_idx = image_datasets["train"].class_to_idx
checkpoint = {
    "arch": args.arch,
    "model": model,
    "learning_rate": args.learning_rate,
    "hidden_units": args.hidden_units,
    "classifier": classifier,
    "epochs": args.epochs,
    "optimizer": optimizer.state_dict(),
    "state_dict": model.state_dict(),
    "class_to_idx": model.class_to_idx,
}

torch.save(checkpoint, "checkpoint.pth")
optimizer.state_dict
