from numpy.core.numeric import indices
from numpy.lib.utils import source
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
from utils.datasets import LoadImages
from utils.general import increment_path
from utils.plots import colors, plot_one_box
import time
from pathlib import Path
from torchvision.ops import nms
from utils.torch_utils import time_sync
from tqdm import tqdm
import platform
import argparse

def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'SSD 🚀 torch {torch.__version__} '  # string
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
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, iou_thres=0.45):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    indices = nms(prediction[0]['boxes'], prediction[0]['scores'], iou_threshold=iou_thres)
    output = prediction.copy()
    output[0]['labels'] = prediction[0]['labels'][indices]
    output[0]['boxes'] = prediction[0]['boxes'][indices]
    output[0]['scores'] = prediction[0]['scores'][indices]
    return output
    
def showbbox(model, path, img, im0s, save_dir, device, merge_img, save_img, save_txt, dataset, merge_txt_path, iou_thres):
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
        pred = model([img.to(device)])
    prediction = non_max_suppression(pred, iou_thres)
    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
    p = Path(p)  # to Path 
    save_path = str(save_dir / p.name)  # img.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    gn_m = torch.tensor(merge_img.shape)[[1, 0, 1, 0]]  # normalization merge image gain whwh   

    img = img.permute(1,2,0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img)  # tensor → ndarray
    
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        if abs(xmax - xmin) < 1 or abs(ymax - ymin) < 1:
            continue
        xyxy = [xmin, ymin, xmax, ymax]

        img_id = os.path.splitext(p.name)[0]
        ys = int(img_id.split('_')[0])
        xs = int(img_id.split('_')[1])
        xyxy_merge = [xyxy[0]+xs*im0.shape[0], xyxy[1]+ys*im0.shape[1], xyxy[2]+xs*im0.shape[0], xyxy[3]+ys*im0.shape[1]]

        label = prediction[0]['labels'][i].item()
        if save_img:
            c = int(label)
            plot_one_box(xyxy, im0, label='destroyed', color=colors(200, True), line_thickness=3)
            plot_one_box(xyxy_merge, merge_img, label='destroyed', color=colors(200, True), line_thickness=3)
        # if label == 1:
        #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
        #     cv2.putText(img, 'mark_type_1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
        #                        thickness=2)
        # elif label == 2:
        #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
        #     cv2.putText(img, 'mark_type_2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
        #                        thickness=2)
        if save_txt:  # Write to file
            xyxy = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # normalized xywh
            line = (label, *xyxy)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

            xywh_m = (torch.tensor(xyxy_merge).view(1, 4) / gn_m).view(-1).tolist()
            line_m = (label, *xywh_m)  # label format
            with open(merge_txt_path + '/' + 'merge.txt', 'a') as f:
                f.write(('%g ' * len(line_m)).rstrip() % line_m + '\n')
    if save_img:
        cv2.imwrite(save_path, im0)
    
def main(opt):
    print('====================begin test====================')
    model = torch.load(opt.weights)
    device = select_device(opt.device)
    model.to(device)

    save_dir = increment_path(Path(opt.project) / opt.name)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    dataset = LoadImages(opt.source, img_size=opt.imgsz, stride=64)
    merge_img = cv2.imread(str(opt.source)+'.tif')  
    merge_txt_path = str(save_dir / 'labels/')

    t1 = time_sync()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img.copy()).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        showbbox(model, path, img, im0s, save_dir=save_dir, device=device, merge_img=merge_img, save_img=opt.save_img,
                 save_txt=opt.save_txt, dataset=dataset, merge_txt_path=merge_txt_path, iou_thres=opt.iou_thres)
    t2 = time_sync()

    if opt.save_img:
        print('saving merge .tif')
        cv2.imwrite(str(save_dir / 'merge.tif'), merge_img) # save the merge image result
        print(f'time cost{t2 - t1:.3f}s')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/mnt/e/github/yolov5-master/runs/train/Faster-Rcnn-mobilenetv32/last.pkl', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/mnt/e/datasets/test_tif/201005069', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1000, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-img', default=True, action='store_true', help='save img results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect/Faster-Rcnn-mobilenetv3', help='save results to project/name')
    parser.add_argument('--name', default='Satellite', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)