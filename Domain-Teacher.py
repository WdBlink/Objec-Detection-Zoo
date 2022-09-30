import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
from utils.datasets import VOCDataset
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torchvision
# from models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, fasterrcnn_transformer_fpn
import argparse
from models import create_model
from options.train_options import TrainOptions

def get_object_detection_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 3 class (mark_type_1，mark_type_2) + background

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def showbbox(model, img, result_path, device):
    # 输入的img是0-1范围的tensor
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img).copy()  # tensor → ndarray

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())

        label = prediction[0]['labels'][i].item()

        if label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            cv2.putText(img, 'mark_type_1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        thickness=2)
        elif label == 2:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(img, 'mark_type_2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        thickness=2)
    cv2.imwrite(result_path, img)


def main(opt):
    root = opt.root
    satellite_path = '/mnt/e/Dataset/aerial_0415'

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 3 classes, mark_type_1，mark_type_2，background
    num_classes = 2
    # use our dataset and defined transformations
    dataset = VOCDataset(root, get_transform(train=True))
    dataset_test = VOCDataset(root, get_transform(train=False))
    dataset_satellite_test = VOCDataset(satellite_path, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    dataset_satellite_test = torch.utils.data.Subset(dataset_satellite_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.rcnn_batch_size, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opt.rcnn_batch_size, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_satellite_test = torch.utils.data.DataLoader(
        dataset_satellite_test, batch_size=opt.rcnn_batch_size, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    # model = fasterrcnn_transformer_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=False)
    model = get_object_detection_model(num_classes)
    model_gan = create_model(opt)  # create a model given opt.model and other options
    model_gan.setup(opt)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=0.0003,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for epochs
    num_epochs = 50
    save_path = './runs/train/DomainTeacher/faster_rcnn.pkl'
    result_path = './runs/train/DomainTeacher/faster_rcnn.jpg'

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, model_gan, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        evaluate(model, data_loader_satellite_test, device=device)

        # save model file
        print(f'save model to {save_path}')
        torch.save(model, save_path)
        print('==================================================')
        print('')

    # print("That's it!")
    # print('==================================================')
    print('====================begin test====================')
    model = torch.load(save_path)
    print(device)
    model.to(device)
    # evaluate(model, data_loader_test, device=device)
    # dataset_test = VOCDataset(root, get_transform(train=False))
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    img, _ = dataset_satellite_test[0]
    showbbox(model, img, result_path=result_path, device=device)

if __name__ == "__main__":
    args = TrainOptions().parse()
    print("Command Line Args:", args)
    main(args)