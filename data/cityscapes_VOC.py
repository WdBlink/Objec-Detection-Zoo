# -* coding: utf-8 -*-
"""
-------------------------------------------------
   Description :   An python example of preparing data for da-faster-rcnn, the exmample is based on adaptation from Cityscapes
   to Foggy Cityscapes, other datasets can be prepared similarly.
   Author :       kinredon
   date：          2019/2/26
-------------------------------------------------
"""
import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from xml.dom.minidom import Document


def writexml(filename, img_shape, bboxes, categories, instance_semantics, xmlpath):
    """

    write to xml style of VOC dataset
    :param filename: xml filename
    :param img_shape: the image shape [H, W, C]
    :param bboxes: bounding boxes
    :param categories: category id
    :param instance_semantics: category id corresponding semantic
    :param xmlpath: xml file save path
    :return: None
    """
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('Cityscapes')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('Cityscapes'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinredon'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(img_shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(instance_semantics[categories[i] - 1]))
        objects.appendChild(object_name)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

    with open(xmlpath, 'w') as f:
        f.write(doc.toprettyxml(indent=' '))


# specify path
source_data_dir = '/mnt/e/datasets/cycle_datasets/cityscape'
target_data_dir = '/mnt/e/datasets/cycle_datasets/foggy_cityscape'

# initialization
img_dir = 'VOCdevkit2007/VOC2007/JPEGImages'
sets_dir = 'VOCdevkit2007/VOC2007/ImageSets/Main'
annotation_dir = 'VOCdevkit2007/VOC2007/Annotations'

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if not os.path.exists(sets_dir):
    os.makedirs(sets_dir)

if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)

# organize images & prepare split list.

# process source train images
source_img_path = os.path.join(source_data_dir, 'leftImg8bit', 'train')
source_train_list = []
print('process source train images')
for root, dirs, files in os.walk(source_img_path, topdown=True):
    files = [name for name in files if name.endswith('.png')]
    for name in tqdm(files):
        im_name = name.replace('.png', '')
        im_name = 'source_' + im_name
        img = cv2.imread(os.path.join(root, name))
        cv2.imwrite(os.path.join(img_dir, im_name + '.jpg'), img)
        source_train_list.append(im_name)

# process target train images.
print('process target train images')
target_img_path = os.path.join(target_data_dir, 'leftImg8bit_foggy', 'train')
target_train_list = []
for root, dirs, files in os.walk(target_img_path, topdown=True):
     files = [ name for name in files if name.endswith('_foggy_beta_0.02.png')]
     for name in tqdm(files):

        im_name = name.replace('.png', '')
        im_name = 'target_' + im_name

        img = cv2.imread(os.path.join(root, name))
        cv2.imwrite(os.path.join(img_dir, im_name + '.jpg'), img)
        target_train_list.append(im_name)


# process target test images
print('process target test images')
target_img_path = os.path.join(target_data_dir, 'leftImg8bit_foggy', 'val')
target_test_list = []
for root, dirs, files in os.walk(target_img_path, topdown=True):
    files = [name for name in files if name.endswith('_foggy_beta_0.02.png')]
    for name in tqdm(files):

            im_name = name.replace('.png', '')
            im_name = 'target_' + im_name

            img = cv2.imread(os.path.join(root, name))
            cv2.imwrite(os.path.join(img_dir, im_name + '.jpg'), img)
            target_test_list.append(im_name)

# write the list
train_list = source_train_list + target_train_list
test_list = target_test_list

with open(os.path.join(sets_dir, "trainval.txt"), 'w') as f:
    for item in train_list:
        f.write("{}\n".format(item))

with open(os.path.join(sets_dir, "test.txt"), 'w') as f:
    for item in test_list:
        f.write("{}\n".format(item))


# prepare the annotation needed for training/testing.
cityscapes_semantics = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
                        'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                        'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                        'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
instance_semantics = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

bind = {}
for i, elt in enumerate(instance_semantics):
    if elt not in bind:
        bind[elt] = i
lb_filter = [bind.get(itm, -1) for itm in cityscapes_semantics]



# instanceIds.png
source_img_path = os.path.join(source_data_dir, 'gtFine')
for root, dirs, files in os.walk(source_img_path, topdown=True):
    files = [name for name in files if name.endswith('instanceIds.png')]
    for name in tqdm(files):

        im_name = name.replace('_gtFine_instanceIds.png', '_leftImg8bit')
        im_name = 'source_' + im_name

        im_inst = cv2.imread(os.path.join(root, name), cv2.IMREAD_ANYDEPTH)
        im_lb = cv2.imread(os.path.join(root, name.replace('_gtFine_instanceIds.png', '_gtFine_labelIds.png')), cv2.IMREAD_ANYDEPTH)

        all_inst_id = np.setdiff1d(np.unique(im_inst), 0)
        boxes = []
        categories = []
        for i_inst in all_inst_id:
            inst_mask = (im_inst == i_inst)
            inst_mask_int = inst_mask - 0
            # print(np.unique(inst_mask))
            assert (len(np.unique(inst_mask_int)) == 2)

            x_cods = np.where(np.sum(inst_mask_int, 0) > 0)
            y_cods = np.where(np.sum(inst_mask_int, 1) > 0)
            box = [np.min(x_cods), np.min(y_cods), np.max(x_cods), np.max(y_cods)]
            boxes.append(box)
            # print(box)
            inst_lb = np.unique(im_lb[inst_mask])
            # print(inst_lb)
            category = lb_filter[inst_lb[0] - 1]
            categories.append(category + 1)
            # break

        # plus 1, in order to match matlab code
        boxes = np.array(boxes) + 1
        categories = np.array(categories)
        boxes = boxes[categories != 0]
        categories = categories[categories != 0]
        filename = im_name + '.jpg'
        img_shape = (*im_inst.shape, 3)
        writexml(filename, img_shape, boxes, categories, instance_semantics,
                 os.path.join(annotation_dir, "{}.xml".format(im_name)))

all_target_images = target_train_list + target_test_list
for im_name in all_target_images:
    source_name = (im_name.replace('_foggy_beta_0.02', '')).replace('target', 'source')
    shutil.copyfile(os.path.join(annotation_dir, "{}.xml".format(source_name)),
                    os.path.join(annotation_dir, "{}.xml".format(im_name)))
print("done!")
# you need move generated 'VOCdevkit2007' directory into $FRCN_ROOT/data  






