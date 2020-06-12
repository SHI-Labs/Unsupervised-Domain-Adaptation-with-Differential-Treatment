import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMultiFeature
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SAVE_PATH = './target_ssl_gt'

IGNORE_LABEL = 255
NUM_CLASSES = 19
RESTORE_FROM = './snapshots/BestGTA5.pth'
SET = 'train'

MODEL = 'Deeplab'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice Deeplab.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = DeeplabMultiFeature(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    if list(saved_state_dict.keys())[0].split('.')[0] == 'module':
        for key in saved_state_dict.keys():
            saved_state_dict['.'.join(key.split('.')[1:])] = saved_state_dict.pop(key)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)
    model.eval()

    trainset = cityscapesDataSet(args.data_dir, args.data_list, 
                                 crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set)
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)

    predicted_label = np.zeros((len(trainset), 512, 1024))
    predicted_prob = np.zeros((len(trainset), 512, 1024))
    image_name = []

    for index, batch in enumerate(trainloader):
        if index % 100 == 0:
            print('%d processd' % index)

        image, _, name = batch
        image = image.to(device)
        with torch.no_grad():
            _, output = model(image)
        output = F.softmax(output, dim=1)
        output = interp(output).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])

    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    print(thres)
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print(thres)
    for index in range(len(trainset)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[(prob<thres[i])*(label==i)] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (args.save, name)) 


if __name__ == '__main__':
    main()
