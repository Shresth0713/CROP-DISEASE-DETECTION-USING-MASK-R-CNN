import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import tensorflow as tf
import imgaug

# Root directory of the project
ROOT_DIR = "D:\minor"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + car and truck

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "potato_healthy")
        self.add_class("object", 2, "potato_late_blight")
        self.add_class("object", 3, "potato_earlyblight")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open('D:\\minor\labels\potato_train.json'))
        # print(annotations1)
        annotations = annotations1

        # Add images and annotations
        for img in annotations:
            image_path = os.path.join(dataset_dir, img["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            polygons = []
            num_ids = []
            for region in img["regions"]:
                if region["shape_attributes"]["name"] == "polygon":
                    polygons.append(region["shape_attributes"])
                    objects = region["region_attributes"]["Disease"]
                    if objects == "potato_healthy":
                        num_ids.append(1)
                    elif objects == "potato_late_blight":
                        num_ids.append(2)
                    elif objects == "potato_earlyblight":
                        num_ids.append(3)

            self.add_image(
                "object",
                image_id=img["filename"],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_custom2(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "potato_healthy")
        self.add_class("object", 2, "potato_late_blight")
        self.add_class("object", 3, "potato_earlyblight")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open('D:\\minor\labels\potato_test.json'))
        # print(annotations1)
        annotations = annotations1

        # Add images and annotations
        for img in annotations:
            image_path = os.path.join(dataset_dir, img["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            polygons = []
            num_ids = []
            for region in img["regions"]:
                if region["shape_attributes"]["name"] == "polygon":
                    polygons.append(region["shape_attributes"])
                    objects = region["region_attributes"]["Disease"]
                    if objects == "potato_healthy":
                        num_ids.append(1)
                    elif objects == "potato_late_blight":
                        num_ids.append(2)
                    elif objects == "potato_earlyblight":
                        num_ids.append(3)

            self.add_image(
                "object",
                image_id=img["filename"],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1
            mask[rr, cc, i] = 1
            # print(len(mask))

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids  # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

#@tf.function
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:\minor\potato", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom2("D:\minor\potato", "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',augmentation = imgaug.augmenters.Sequential([
                imgaug.augmenters.Fliplr(1),
                imgaug.augmenters.Flipud(1),
                imgaug.augmenters.Affine(rotate=(-45, 45)),
                imgaug.augmenters.Affine(rotate=(-90, 90)),
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images

                ]

                ))


config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
# Download weights file
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

train(model)