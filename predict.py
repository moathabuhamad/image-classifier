# Imports here
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store", default="checkpoint.pth")
    parser.add_argument("--top_k", dest="top_k", default="3")
    parser.add_argument(
        "--filepath", dest="filepath", default="flowers/test/1/image_06743.jpg"
    )  # use a deafault filepath to a primrose image
    parser.add_argument(
        "--category_names", dest="category_names", default="cat_to_name.json"
    )
    parser.add_argument("--gpu", action="store", default="gpu")
    return parser.parse_args()


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    im = Image.open(image)
    image_size = im.size
    width = image_size[0]
    height = image_size[1]

    if width > height:
        im.thumbnail(
            (5000000, 256)
        )
    else:
        im.thumbnail(
            (256, 5000000)
        )

    left_margin = (im.width - 224) / 2
    bottom_margin = (im.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    im = im.crop((left_margin, bottom_margin, right_margin, top_margin))
    np_image = np.array(im)

    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nor_image = (np_image - mean) / std

    final_image = nor_image.transpose((2, 0, 1))

    return final_image


def predict(image_path, model, topk):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    model.to(
        device
    )

    image = process_image(image_path)
    torch_image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    torch_image_tensor.unsqueeze_(
        0
    )
    torch_image_tensor = torch_image_tensor.to(
        device
    )

    with torch.no_grad():
        log_ps = model(torch_image_tensor)

        ps = torch.exp(log_ps)
        probs, labels = ps.topk(3, dim=1)

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    probs = probs.cpu().numpy()[0]
    classes = []
    for label in labels.cpu().numpy()[0]:
        classes.append(idx_to_class[label])

    print(probs)
    print(classes)
    return probs, classes


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg13(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.load_state_dict = checkpoint["state_dict"]
    model.class_to_idx = checkpoint["class_to_idx"]
    learning_rate = checkpoint["learning_rate"]
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])

    return optimizer, model


def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names


def main():
    args = parse_args()
    gpu = args.gpu
    optimizer, model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)

    img_path = args.filepath
    probs, classes = predict(img_path, model, args.top_k)
    labels = [cat_to_name[index] for index in classes]

    print(labels)
    print(probs)


if __name__ == "__main__":
    main()
