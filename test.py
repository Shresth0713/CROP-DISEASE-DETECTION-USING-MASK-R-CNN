import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#import custom

# Root directory of the project
ROOT_DIR = "D:\minor"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = "D:\minor\logs\object20230509T0150\mask_rcnn_object_0020.h5"   # change it
class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 10  # Background + Car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        self.add_class("object", 1, "potato_healthy")
        self.add_class("object", 2, "potato_late_blight")
        self.add_class("object", 3, "potato_earlyblight")
        self.add_class("object", 4, "Bacterial_Blight_Rice")
        self.add_class("object", 5, "Blast_Rice")
        self.add_class("object", 6, "Brown_Spot_Rice")
        self.add_class("object", 7, "TUNGRO_Rice")
        self.add_class("object", 8, "Healthy_Wheat")
        self.add_class("object", 9, "septoria_wheat")
        self.add_class("object", 10, "stripe_rust_wheat")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(
            open('D:\\minor\labels\output2.json'))

        annotations = annotations1   # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['Disease'] for s in a['regions']]
            print("objects:", objects)
            name_dict = {"potato_healthy": 1, "potato_late_blight": 2, "potato_earlyblight": 3,
                         "Bacterial_Blight_Rice": 4, "Blast_Rice": 5, "Brown_Spot_Rice": 6, "TUNGRO_Rice": 7,
                         "Healthy_Wheat": 8, "septoria_wheat": 9, "stripe_rust_wheat": 10}
            num_ids = [name_dict[a] for a in objects]

            print("numids", num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

TEST_MODE = "inference"
ROOT_DIR = "D:\minor\dataset"

def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in all visualizations in the notebook.  Provide a central point to control graph sizes. Adjust the size attribute to control how big to render images"""
        _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        return ax

        # Load validation dataset
        # Must call before using the dataset
CUSTOM_DIR = "D:\minor\dataset"
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

config = CustomConfig()
    # LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

weights_path = WEIGHTS_PATH
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# This is for predicting images which are not present in dataset
path_to_new_image = 'D:\\minor\\test\\lolr(208).jpg'
image1 = mpimg.imread(path_to_new_image)

# Run object detection
#print(len([image1]))
results1 = model.detect([image1], verbose=1)

# Display results
# ax = get_ax(1)
# r1 = results1[0]
# visualize.display_instances(image1, r1['rois'], r1['masks'], r1['class_ids'],
# dataset.class_names, r1['scores'], ax=ax, title="Predictions1")
#
# image_id = random.choice(dataset.image_ids)
# #image_id = 'D:/MaskRCNN-aar/Dataset/val/1.jfif'
# print("image id is :",image_id)
# image, image_meta, gt_class_id, gt_bbox, gt_mask =\
# modellib.load_image_gt(dataset, config, image_id)
# info = dataset.image_info[image_id]
# print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,dataset.image_reference(image_id)))
#
# # Run object detection
# results = model.detect([image], verbose=1)
#
# x = get_ax(1)
# r = results[0]
# ax = plt.gca()
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
#
#
# total_gt = np.array([])
# total_pred = np.array([])
# mAP_ = []  # mAP list
#
# # compute total_gt, total_pred and mAP for each image in the test dataset
# # Compute total ground truth boxes(total_gt) and total predicted boxes(total_pred) and mean average precision for each Image
# # in the test dataset
# for image_id in dataset.image_ids:
#     image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self.dataset, self.inference_model.config,image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(mold_image(image, self.inference_model.config), 0)
#     results = self.inference_model.detect(molded_images, verbose=0)
#     r = results[0]
#             # Compute mAP - VOC uses IoU 0.5
#     AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
#                                            r["class_ids"], r["scores"], r['masks'])
#     mAPs.append(AP)
#     print("Average precision of this image : ", AP_)
#     print("The actual mean average precision for the whole images", sum(mAP_) / len(mAP_))

# import pandas as pd
#
# total_gt = total_gt.astype(int)
# total_pred = total_pred.astype(int)
# # save the vectors of gt and pred
# save_dir = "output"
# gt_pred_tot_json = {"Total Groundtruth": total_gt, "predicted box": total_pred}
# df = pd.DataFrame(gt_pred_tot_json)
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# df.to_json(os.path.join(save_dir, "gt_pred_test.json"))