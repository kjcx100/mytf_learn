#!/usr/bin/env python3

import datetime
import json
import csv
import os
import cv2
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

csv_file=csv.reader(open("./testtrain_1w.csv"))
imgRoot="./train_1w"
ROOT_DIR = 'train_1w'
IMAGE_DIR = ROOT_DIR #os.path.join(ROOT_DIR, "shapes_train2018")
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'car',
        'supercategory': '',
    },
]


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    number = 0
    for iterm in csv_file:
        if number != 0:
            image_name = iterm[0]
            # print (image_name)
            imgPath = os.path.join(imgRoot, image_name)
            im = cv2.imread(imgPath)
            try:
                im.shape
            except:
                print('fail to read xxx.jpg')
                continue
            image = Image.open(imgPath)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(imgPath), image.size)
            coco_output["images"].append(image_info)
            print("image_info:",image_info)

            #从文件中解析出多个标记
            rects = iterm[1]
            if len(rects) < 1:
                print('rects<1 so continue !!!')
                continue
            rects = rects.split(";")

            for i in range(0, len(rects)):
                rect = rects[i]
                if len(rects[i]) < 3:
                    print('rects[i] < 3 so continue !!!')
                    continue
                rect = rect.split("_")
                # if float(rect[0])< 0:
                #    print('rect[0]< 0')
                #    continue
                # print (rect)
                xmin = int(float(rect[0]))
                ymin = int(float(rect[1]))
                w = int(float(rect[2]))
                h = int(float(rect[3]))
                rect_int = [xmin,ymin,w,h]
                class_id = 1
                category_info = {'id': class_id, 'is_crowd': 0}
                annotation_info = pycococreatortools.create_yolo_annotation_info(
                    segmentation_id, image_id, category_info, rect_int,image.size)

                print("annotation_info:", annotation_info)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id += 1
                # lxl add
                #height, width, channels = im.shape
                # value = Is_RectOutRange((width,height),(xmin,ymin,w,h))
                value = 0
                if value < 0:
                    print('RectOutRange so continue !!!')
                    continue
                ####cv2.rectangle(im,(xmin,ymin),(xmin+w,ymin+h),(0,255,0),1)
            ############################

            ####cv2.imshow("im",im)
            ####cv2.waitKey(0)
            number += 1
            image_id += 1
        else:
            number += 1
    with open('{}/instances_yolo_shape_train.json'.format('.'), 'w') as output_json_file:
        json.dump(coco_output, output_json_file,indent= 4)
'''
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        print(image_files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
            print("image_info:", image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    # print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
                    print("annotation_filename:", annotation_filename)
                    print("annotation_info:", annotation_info)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    with open('{}/instances_shape_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
'''


if __name__ == "__main__":
    main()
