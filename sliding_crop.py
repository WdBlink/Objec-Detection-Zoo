import argparse
from tqdm import tqdm
from pathlib import Path

import cv2
import os
import glob

from utils.general import colorstr


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
def crop_and_save(img, crop_size, output_path):
    shape = img.shape[:2]
    xs = shape[0] // crop_size
    ys = shape[1] // crop_size
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    with tqdm(total=ys*xs) as pbar:
        for y in range(ys):
            for x in range(xs):
                cropped = img[y*crop_size:(y+1)*crop_size+1, x*crop_size:(x+1)*crop_size, :]
                path = os.path.join(output_path, f'{y}_{x}.tif')
                try:
                    cv2.imwrite(path, cropped)
                    assert 1==1
                except Exception as err:
                    print(err)
                pbar.update(1)

def run(source='',  # file/dir/URL/glob, 0 for webcam
        output='', #
        imgsz=640,  # inference size (pixels)
        ):
    p = str(Path(source).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for path in images:
        img_id = os.path.splitext(os.path.split(path)[1])[0]
        img = cv2.imread(path)
        output_path = os.path.join(output, img_id)
        crop_and_save(img=img, crop_size=imgsz, output_path=output_path)
    return 0

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/datasets/test_tif', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output', type=str, default='/datasets/test_tif', help='path to cropped images')
    parser.add_argument('--imgsz', type=int, default='512', help='cropped images size')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('crop: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)