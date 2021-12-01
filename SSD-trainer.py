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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
from utils.general import colorstr, increment_path
from pathlib import Path
import platform
      
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

def select_device(device='', batch_size=None, model_name='SSD'):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'{model_name} 🚀 torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')

def run(weights='',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        model_name='SSD',
        imgsz=640,  # inference size (pixels)
        iou_thres=0.45,  # NMS IOU threshold
        workers=0,
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        resume=False,
        epochs=50,
        batch_size=16,
        save_txt=True,  # save results to *.txt
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        line_thickness=3,  # bounding box thickness (pixels)
        learning_rate=0.001,
        num_classes=2   # 3 classes, mark_type_1，mark_type_2，background
        ):
    root = source

    # train on the GPU or on the CPU, if a GPU is not available
    device = select_device(device, batch_size=batch_size, model_name=model_name)

    # use our dataset and defined transformations
    dataset = VOCDataset(root, get_transform(train=True))
    dataset_test = VOCDataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    # 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    if model_name == 'Faster-Rcnn-mobilenetv3':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  
    elif model_name == 'Faster-Rcnn-resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    elif model_name == 'SSD':
        model = torchvision.models.detection.ssd300_vgg16(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    elif model_name == 'SSDlite':
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
        
    if resume is True:
        print(f'resuming checkpoint from {weights}')
        model = torch.load(weights)
    # 或get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # Adam
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=learning_rate)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = epochs
    save_dir = str(increment_path(Path(project) / model_name, exist_ok=False))
    root = '/mnt/e/github/yolov5-master/'
    output_path = os.path.join(root, save_dir)
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    save_path = os.path.join(output_path, 'last.pkl')

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset    
        evaluate(model, data_loader_test, device=device)    
        
        # save model file
        print(f'save model to {save_path}')
        torch.save(model, save_path)
        print('==================================================')
        print('')

    print("That's it!")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/mnt/e/github/yolov5-master/runs/train/Faster-Rcnn-mobilenetv32/last.pkl', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/mnt/e/Dataset//taining_data_2021-08-19', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--model-name', type=str, default='Faster-Rcnn-mobilenetv3', help='FasterRcnn, SSD, Mask R-CNN, Keypoint R-CNN, RetinaNet')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1000, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=24, help='total batch size for all GPUs')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default='runs/train', help='save results to project/name')
    parser.add_argument('--name', default='faster-rcnn-mobilenetv3', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--learning-rate', default=0.01, type=int, help='learning rate of optimizer')
    parser.add_argument('--num-classes', default=2, type=int, help='the number of classes')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)