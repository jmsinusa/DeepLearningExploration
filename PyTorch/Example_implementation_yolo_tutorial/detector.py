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

from PIL import Image, ImageDraw, ImageFont


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

        # Move to GPU if available (or if use_cuda=True)
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = self.model.cuda()

        # Not sure if this is needed.
        self.model.eval()

        # Run config to set default values
        self.config()

        # Load class names
        if class_names:
            self.load_class_names(class_names=class_names)

        # Assign colour map
        self.config_params['colour_map'] = {'person': (198, 0, 0),
                           'bicycle': (164, 125, 70),
                           'car': (107, 76, 36),
                           'motorbike': (164, 125, 70),
                           'aeroplane': (92, 92, 92),
                           'bus': (69, 48, 23),
                           'train': (122, 41, 118),
                           'truck': (69, 48, 23),
                           'boat': (69, 48, 23),
                           'stop sign': (122, 41, 118),
                           'parking meter': (122, 41, 118),
                           'bird': (196, 215, 136),
                           'cat': (176, 172, 59),
                           'dog': (176, 172, 59),
                           'horse': (122, 41, 118),
                           'sheep': (122, 41, 118),
                           'cow': (122, 41, 118),
                           'elephant': (122, 41, 118),
                           'bear': (122, 41, 118),
                           'zebra': (122, 41, 118),
                           'giraffe': (122, 41, 118),
                           'backpack': (117, 157, 209),
                           'skateboard': (164, 125, 70),
                           'toilet': (122, 41, 118),
                           'suitcase': (57, 90, 172),
                           'tie': (57, 90, 172)}

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

    def config(self, output_folder=r'/data/detection/output', batch_size=2, min_confidence=0.5,
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
        loader = DataLoader(img_dataset, batch_size=self.config_params['batch_size'], shuffle=True,
                            collate_fn=collate_fn)
        # Now work through batches
        for batch in loader:
            input_imgs = batch['trans_image']
            if self.use_cuda:
                input_imgs = input_imgs.cuda()

            # print('MODEL', self.model)
            prediction = self.model(input_imgs, self.use_cuda)
            preds = util.write_results(prediction, self.config_params['min_confidence'],
                                       self.config_params['num_classes'],
                                       nms_conf=self.config_params['nms_iou_thresh'])
            for indx in range(len(batch['trans_image'])):
                # Now check to see if any images have been returned

                this_pred = preds[preds[:, 0] == indx].cpu()
                this_pred = this_pred.numpy()
                if this_pred.shape[0] == 0:
                    print("No detections for image", batch["img_filename"][indx])
                    continue

                # Now do something
                print("This pred contains", this_pred.shape[0], "objects")
                # Look up the classes detected
                classes = this_pred[:, 7]
                classes = classes
                classes = classes.astype(np.int8)
                classes_str = [self.class_names[class_] for class_ in classes]
                classes_unique = list(set(classes_str))
                # print('Classes in this image:', list(set(classes_str)))

                # Convert this_pred coordinates into original size
                original_size = batch['original_image_size'][indx]
                max_orig_size = np.max(original_size)
                padding = batch['padding'][indx]

                x1 = this_pred[:, 1]
                y1 = this_pred[:, 2]
                x2 = this_pred[:, 3]
                y2 = this_pred[:, 4]

                x1 = _resize_coords(x1, self.config_params['input_res'], max_orig_size)
                x2 = _resize_coords(x2, self.config_params['input_res'], max_orig_size)
                y1 = _resize_coords(y1, self.config_params['input_res'], max_orig_size)
                y2 = _resize_coords(y2, self.config_params['input_res'], max_orig_size)
                x1, x2, y1, y2 = _translate_coords(x1, x2, y1, y2, padding)
                x1, x2, y1, y2 = _crop_to_original_size(x1, x2, y1, y2, original_size)

                # Put these back into this_pred (OPTIONAL)
                this_pred[:, 1] = x1
                this_pred[:, 2] = y1
                this_pred[:, 3] = x2
                this_pred[:, 4] = y2

                # Recall original image
                img = batch['original_image'][indx]

                #### SET OUT_FILEPATH

                #### If dog, save in "dog" directory
                dir_class = None
                if 'dog' in classes_str:
                    dir_class = 'dog'
                elif 'person' in classes_str:
                    dir_class = 'person'
                elif 'car' in classes_str:
                    dir_class = 'car'
                if not dir_class:
                    for class_ in ['bicycle', 'motorbike', 'bus', 'truck']:
                        if class_ in classes_str:
                            dir_class = 'other_transport'
                if not dir_class:
                    dir_class = classes_unique[0].replace(' ', '_')
                    # try:
                    #     dir_class = classes_unique[0].replace(' ','_')
                    # except TypeError:
                    #     print("ERROR selecting dir_class")
                    #     print(classes_unique)
                    #     print("----------------------------")
                    #     print(batch['img_filename'][indx][0])
                # Prepare output dir
                output_dir = os.path.join(self.config_params['output_folder'], dir_class)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                filename = os.path.splitext(batch['img_filename'][indx])[0] + '.png'
                output_filepath = os.path.join(output_dir, filename)

                print(output_filepath)

                draw_output_images(img, this_pred, classes_str, self.config_params['colour_map'], output_filepath)



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


def draw_output_images(img, predictions, classes_str, colour_map, output_filepath):
    """
    Draw
    :param img: PIL image
    :param predictions: predictions, n x 8
            0: batch pos (ignore), 1: x1, 2: y1, 3: x2, 4: y2, 5: box confidence, 6: class confidence, 7: class no.
    :param classes_str: List of class stringss
    :param class_colours: Dict relating class string to box colour. If not in, use 'default'
    :param output_filepath: Path to save produced image to.
    :return: None
    """
    # Get coordiantes
    x1 = predictions[:, 1]
    y1 = predictions[:, 2]
    x2 = predictions[:, 3]
    y2 = predictions[:, 4]
    boxconf = predictions[:, 5]
    classconf = predictions[:, 6]

    draw = ImageDraw.Draw(img)
    for ii in range(x1.shape[0]):
        points = (x1[ii], y1[ii]), (x2[ii], y1[ii]), (x2[ii], y2[ii]), (x1[ii], y2[ii]), (x1[ii], y1[ii])

        this_class = classes_str[ii]
        try:
            colour = colour_map[this_class]
        except KeyError:
            colour = (138, 138, 138)
        #         draw.rectangle((x1[ii],y1[ii],x2[ii],y2[ii]), fill=None, outline=colour)
        draw.line(points, fill=colour, width=3)
        text = "%s; %4.3f; %4.3f)" % (this_class, boxconf[ii], classconf[ii])
        #         font=ImageFont.load_default()
        font = ImageFont.truetype(font=r'cfg/FreeSansBold.ttf', size=13)
        text_size = font.getsize(text)
        draw.rectangle((x1[ii] + 1, y1[ii] + 1, x1[ii] + 1 + text_size[0], y1[ii] + 1 + text_size[1]), fill="black")
        draw.text((x1[ii], y1[ii]), text, fill='white', font=font)

    img.save(output_filepath)

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
            samples[key].append(sample[key])
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
            padding = (diff1, 0, diff2, 0)  # left, top, right and bottom
        else:
            # pad height
            padding = (0, diff1, 0, diff2)

        # transforms.Pad(padding, fill=self.padding_val, padding_mode='constant')
        image = transforms.functional.pad(image, padding, fill=self.padding_val, padding_mode='constant')

        return image, padding


def _resize_coords(coord, trans_size, orig_size):
    factor = float(orig_size) / float(trans_size)
    return np.round(coord * factor).astype(np.int16)


def _translate_coords(x1, x2, y1, y2, padding):
    left, top, right, bottom = padding
    x1 -= left
    x2 -= left
    y1 -= top
    y2 -= top
    return x1, x2, y1, y2


def _crop_to_original_size(x1, x2, y1, y2, orig_size):
    x1[x1 < 0] = 0
    x2[x2 > orig_size[0]] = orig_size[0]
    y1[y1 < 0] = 0
    y2[y2 > orig_size[1]] = orig_size[1]
    return x1, x2, y1, y2
