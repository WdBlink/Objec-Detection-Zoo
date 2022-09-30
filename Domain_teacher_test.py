import torch
import os
import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
from utils.datasets import VOCDataset, DomainDataset
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torchvision
from models.detection.faster_rcnn import FastRCNNPredictor
from models.domainnet import DomainNet
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_transformer_fpn
import argparse
from torch.nn import MSELoss


weight_path = "E:\\github\\object-detection-zoo\\runs\\train\\domain_net.pkl"
domain_net = torch.load(weight_path)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
domain_net.to(device)

root = "E:/Dataset/taining_data_2021-08-19"
dataset_test = DomainDataset(root)
# split the dataset in train and test set
indices = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
# define training and validation data loaders
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

for ims, targets, dims in data_loader_test:
    images = list(image.to(device) for image in ims)
    domain_images = list(image.to(device) for image in dims)

    img_stack = torch.stack(images)
    dim_stack = torch.stack(domain_images)

    input = torch.concat((img_stack, dim_stack), dim=1)
    output = domain_net(input)