import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import pdb
import cv2
#from scipy.misc import imresize

class synthiaDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/%s" % name)
            label_file = osp.join(self.root, "synthia_mapped_to_cityscapes/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        #labelcv = cv2.imread(datafiles["label"], cv2.IMREAD_UNCHANGED)[:,:,2].astype(np.float32)
        #print(label.shape, labelcv.shape)
        #assert label == labelcv
        #pdb.set_trace()
        #label = imresize(label[:,:,2], self.crop_size, interp='nearest').astype(np.float32)
        #for i in range(23):
        #    print(np.sum(label==i))
        #pdb.set_trace()
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    dst = synthiaDataSet("./data/SYNTHIA", "./dataset/synthia_list/train.txt", crop_size=(1280,760))
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, _,_ = data
        for j in range(13):

            print(j, np.sum((labels == j).numpy()))
        pdb.set_trace()
        plt.imshow(np.moveaxis(imgs.int().numpy()[0,:,:,:]+128, 0, -1))
        plt.show()
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
    print(i)
