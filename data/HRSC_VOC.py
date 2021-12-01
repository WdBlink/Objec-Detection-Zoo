import shutil
import xml.etree.ElementTree as ET
import os
from PIL.Image import ImagePointHandler

from tqdm import tqdm

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
    w = int(root.find('Img_SizeWidth').text)
    h = int(root.find('Img_SizeHeight').text)

    for obj in root.find('HRSC_Objects').iter('HRSC_Object'):
        bb = convert_box((w, h), [float(obj.find(x).text) for x in ('box_xmin', 'box_xmax', 'box_ymin', 'box_ymax')])
        cls_id = int(obj.find('Class_ID').text) % 100 # class id
        out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

# Convert
path = '/mnt/e/datasets/HSRC2016/HRSC2016'
for year, image_set in ('ship_detection_', 'train'), ('ship_detection_', 'val'), ('ship_detection_', 'trainval'), ('ship_detection_', 'test'):
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
        f = f'{path}/JPEGImages/{id}.bmp'  # old img path
        # lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
        lb_path = os.path.join(lbs_path, f"{id}.txt")
        shutil.copyfile(f, f'{imgs_path}/{id}.bmp')
        convert_label(path, lb_path, year, id)  # convert labels to YOLO format