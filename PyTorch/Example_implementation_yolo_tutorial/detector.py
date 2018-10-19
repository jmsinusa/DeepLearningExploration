"""
The module that will be used to access the trained YOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import util  # Our helpful util functions.
import darknet
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# from skimage import io
# from skimage.transform import resize

from PIL import Image


class YOLO(object):
    """
    Class for using pretrained YOLO
    """

    def __init__(self, config_file="cfg/yolov3.cfg", weights=r'/data/pretrained/yolov3/yolov3.weights',
                 use_cuda=torch.cuda.is_available(), class_names=r'/data/pretrained/yolov3/coco.names'):
        """
        Initialise class; loads config file; loads weights (unless None supplied)
        :param config_file: Offical config
        :param weights: Weight file
        :param use_cuda: Defaults to True if CUDA available.
        """
        self.model = None  # To store model
        self.class_names = None  # To store class names
        self.config_params = {}

        self.load_model(config_file=config_file)
        if weights:
            self.load_weights(weights=weights)
        # Move to GPU if available (or if use_cuda=True
        if use_cuda:
            self.model = self.model.cuda()

        # Not sure if this is needed.
        self.model.eval()

        # Run config to set default values
        self.config()

        # Load class names
        if class_names:
            self.load_class_names(class_names=class_names)

    def load_model(self, config_file="cfg/yolov3.cfg"):
        """
        Load model from config
        :param config_file: full or relative path to config.
        :return:
        """
        self.config_params['config_file'] = config_file
        self.model = darknet.Darknet(config_file)

    def load_weights(self, weights=r'/data/pretrained/yolov3/yolov3.weights'):
        """
        Load weights into model.
        :param weights: Path to weights file
        :return:
        """
        self.config_params['weights_files'] = weights
        self.model.load_weights(weights)

    def load_class_names(self, class_names=r'/data/pretrained/yolov3/coco.names'):
        """
        Load names of COCO classes (default) or any others.
        :param class_names: file path to names
        :return:
        """
        with open(class_names, "r") as fid:
            names = fid.read().split("\n")[:-1]
        self.class_names = names
        self.config_params['num_classes'] = len(names)

    def config(self, output_folder=r'/data/detection/output', batch_size=1, min_confidence=0.5,
               nms_iou_thresh=0.4, input_res=608):
        """
        Updates self.config with config given above
        :param output_folder: Directory to store detections
        :param batch_size: Batch size for inference
        :param min_confidence: Minimum detection confidence. All others are removed.
        :param nms_iou_thresh: Maximum IOU for non-maximum suppression.
        :param input_res: Network input resolution. Config file default is 608. Tutorial suggests 416
        :return:
        """
        assert os.path.isdir(output_folder)
        self.config_params['output_folder'] = output_folder
        assert type(batch_size) == int
        self.config_params['batch_size'] = batch_size
        assert 1 >= min_confidence >= 0
        self.config_params['min_confidence'] = min_confidence
        assert 1 >= nms_iou_thresh >= 0
        self.config_params['nms_iou_thresh'] = nms_iou_thresh
        assert type(input_res) == int
        assert input_res % 32 == 0
        assert input_res > 32
        self.config_params['input_res'] = input_res

    def infer_write_dir(self, image_dir):
        """
        The standard method for inferring a directory of images.
        Change output by running self.config(output_folder=<new_folder>)
            or by setting self.config_params['output_folder']
        :param image_dir: Directory to read images from
        :return:
        """
        # Read directory using data loader
        img_dataset = ImagesDataset(image_dir)
        for sample in img_dataset:
            pass
        results = self.infer(images)
        out_filenames = None
        self.write_images(images, results, out_filenames)

    def infer(self, images):
        pass

    def write_images(self, images, results, out_filenames):
        pass

    def infer_write(self, images, out_filenames):
        # For when images are already in memory
        results = self.infer(images)
        out_filenames = None
        self.write_images(images, results, out_filenames)

    def display_image(self, images, result):
        pass

    def infer_display(self, images):
        result = self.infer(images)
        self.display_image(images, result)


class ImagesDataset(Dataset):
    """
    Data set from images in local directory
    """

    def __init__(self, image_dir, img_size=608):
        self.image_dir = image_dir
        self.image_files_list = [s for s in os.listdir(image_dir) if
                                 os.path.splitext(s)[1] in ['.jpg', '.png', '.jpeg']]
        self.trans_makesquare = MakeSquare(0)
        self.trans_resize = transforms.Resize(img_size)
        self.trans_totensor = transforms.ToTensor()
        # self.transform = transforms.Compose([MakeSquare(0), transforms.Resize(img_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        """
        Takes index, returns sample
        :param idx:
        :return: sample = {original_image: original image (PIL), trans_image: transformed image (tensor),
                            padding: padding (left, top, right and bottom) applied to original image to make square,
                            original_image_size: (w, h) of original dims, img_filename: original image filename}
        """
        sample = {}

        img_name = os.path.join(self.image_dir,
                                self.image_files_list[idx])
        sample['img_filename'] = self.image_files_list[idx]
        img = Image.open(img_name)
        sample['original_image'] = img
        sample['original_image_size'] = img.size
        # Transform
        img, padding = self.trans_makesquare(img)
        sample['padding'] = padding
        img = self.trans_resize(img)
        img = self.trans_totensor(img)
        sample['trans_image'] = img
        return sample


def collate_fn(batch):
    """
    Function for collating batches of samples from ImagesDataset
    :param batch: 
    :return: samples dict {original_image: [list of original images as PIL], trans_image: tensor, stacked in first dim,
                          padding: [list of (left, top, right, bottom)], original_image_size: [list of (w, h)],
                          img_filename: [list of img_filenames]}
    """
    samples = {'original_image': [], 'padding': [], 'original_image_size': [], 'img_filename': []}
    trans_images_list = []
    for sample in batch:
        for key in samples.keys():
            samples[key].append(sample['key'])
        trans_images_list.append(sample['trans_image'])
    samples['trans_image'] = torch.stack(trans_images_list, 0, )
    return samples


# ##### Transforms

class MakeSquare(object):
    """
    Pad the image to make square
    """
    def __init__(self, padding_val=0):
        """
        Pad shorter dimension with padding_val
        :param padding_val:
        """
        assert isinstance(padding_val, int)
        self.padding_val = padding_val

    def __call__(self, image):
        """
        Return modified sample[image] padded to square
        :param sample:
        :return:
        """
        padding = (0, 0, 0, 0)
        w, h = image.size
        if h == w:
            # Do nothing
            return image, padding

        diff1 = int(np.abs(np.ceil((w - h) / 2.0)))
        diff2 = int(np.abs(np.floor((w - h) / 2.0)))
        if h > w:
            # pad width
            padding = (diff1, 0, diff2, 0) #  left, top, right and bottom
        else:
            # pad height
            padding = (0, diff1, 0, diff2)

        # transforms.Pad(padding, fill=self.padding_val, padding_mode='constant')
        image = transforms.functional.pad(image, padding, fill=self.padding_val, padding_mode='constant')

        return image, padding

