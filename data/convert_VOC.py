import shutil
import xml.etree.ElementTree as ET
import os
from PIL.Image import ImagePointHandler

from tqdm import tqdm
from utils.general import download, Path

class_names = ['1']

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(f'{path}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in class_names:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = class_names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# Convert
path = '/mnt/e/Dataset/taining_data_2021-08-19'
for year, image_set in ('destroyed_building', 'train'), ('destroyed_building', 'val'), ('destroyed_building', 'trainval'), ('destroyed_building', 'test'):
    # imgs_path = dir / 'images' / f'{image_set}{year}'
    imgs_path = os.path.join(path, 'images', f'{image_set}_{year}')
    # lbs_path = dir / 'labels' / f'{image_set}{year}'
    lbs_path = os.path.join(path, 'labels', f'{image_set}_{year}')
    if os.path.exists(imgs_path):
        pass
    else:
        os.makedirs(imgs_path)
        os.makedirs(lbs_path)
    # imgs_path.mkdir(exist_ok=True, parents=True)
    # lbs_path.mkdir(exist_ok=True, parents=True)
    image_ids = open( f'{path}/ImageSets/Main/{image_set}.txt').read().strip().split()
    for id in tqdm(image_ids, desc=f'{image_set}{year}'):
        f = f'{path}/JPEGImages/{id}.tif'  # old img path
        # lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        lb_path = os.path.join(lbs_path, f"{id}.txt")
        shutil.copyfile(f, f'{imgs_path}/{id}.tif')
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format