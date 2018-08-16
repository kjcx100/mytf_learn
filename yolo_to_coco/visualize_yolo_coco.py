##%matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
import cv2

image_directory = 'train_1w'
annotation_file = './instances_yolo_shape_train.json'
example_coco = COCO(annotation_file)
categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))
category_ids = example_coco.getCatIds(catNms=['car'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]

# load and display instance annotations
#image = io.imread(image_directory + image_data['file_name'])
imgPath = os.path.join(image_directory,image_data['file_name'])
image = io.imread(imgPath)
plt.imshow(image); plt.axis('on')
im = cv2.imread(imgPath)
cv2.imshow("im",im)
cv2.waitKey(0)

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
print(annotations)
example_coco.showAnns(annotations)
