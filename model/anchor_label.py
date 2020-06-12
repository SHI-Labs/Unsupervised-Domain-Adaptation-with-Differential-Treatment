from PIL import Image
import numpy as np
#import cv2 as cv
#import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from multiprocessing import Pool
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.measure import label as sklabel
from skimage.feature import peak_local_max
import queue
import pdb

#FG_LABEL = [6,7,11,12,13,14,15,16,17,18] #light,sign,person,rider,car,truck,bus,train,motor,bike
FG_LABEL = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
BG_LABEL = [0,1,2,3,4,8,9,10]
#w, h= 161, 91
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def save_img(img_path, anchors):
    img = Image.open(img_path).convert('RGB')
    w, h= 1280, 720
    img = img.resize((w,h), Image.BICUBIC)
    
    img = np.asarray(img)
    ih, iw, _ = img.shape
    for i in range(len(FG_LABEL)):
        if len(anchors[i]) > 0:
            color = (palette[3*FG_LABEL[i]], palette[3*FG_LABEL[i]+1], palette[3*FG_LABEL[i]+2])
            for j in range(len(anchors[i])):
                x1, y1, x2, y2, _ = anchors[i][j,:].astype(int)
                #cv2.rectangle(img, (y1, x1), (y2, x2), color, 2)
        imgsave = Image.fromarray(img, 'RGB')
        imgsave.save('sample.png')
        #pdb.set_trace()
        #img=mpimg.imread('sample.png')
        #imgplot = plt.imshow(img)
        #plt.show()


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def anchorsbi(label, origin_size=(720,1280), iou_thresh=0.4):
    h,w = label.shape[0], label.shape[1]
    h_scale, w_scale = float(origin_size[0])/h, float(origin_size[1])/w
    hthre = np.ceil(32./h_scale)
    wthre = np.ceil(32./w_scale)
    anchors = []
    for fg in FG_LABEL:
        mask = label==fg
        foreidx = 1
        if torch.sum(mask)>0:
            
            masknp = mask.cpu().clone().detach().numpy().astype(int)
            seg, foreidx = sklabel(masknp, background=0, return_num=True, connectivity=2)
            foreidx += 1

        anc_cls = np.zeros((foreidx-1,5))
        for fi in range(1, foreidx):
            x,y = np.where(seg==fi)
            anc_cls[fi-1,:4] = np.min(x), np.min(y), np.max(x), np.max(y)
            area = (anc_cls[fi-1,2] - anc_cls[fi-1,0])*(anc_cls[fi-1,3] - anc_cls[fi-1,1])
            anc_cls[fi-1,4] = float(len(x)) / max(area, 1e-5)
        if len(anc_cls) > 0:
            hdis = anc_cls[:,2] - anc_cls[:,0]
            wdis = anc_cls[:,3] - anc_cls[:,1]
            anc_cls = anc_cls[np.where((hdis>=hthre)&(wdis>=wthre))[0],:]
            area = (anc_cls[:,2] - anc_cls[:,0])*(anc_cls[:,3] - anc_cls[:,1])
            sortidx = np.argsort(area)[::-1]
            anc_cls = anc_cls[sortidx,:]
            if len(anc_cls) > 0:
                anc_cls = anc_cls[np.where(anc_cls[:,4]>=iou_thresh)[0],:]
            if len(anc_cls) > 0:
                anc_cls[:,0] = np.floor(h_scale*anc_cls[:,0])
                anc_cls[:,2] = np.ceil(h_scale*anc_cls[:,2])
                anc_cls[:,2] = np.where(anc_cls[:,2]<origin_size[0], anc_cls[:,2], origin_size[0])
                anc_cls[:,1] = np.ceil(w_scale*anc_cls[:,1])
                anc_cls[:,3] = np.ceil(w_scale*anc_cls[:,3])
                anc_cls[:,3] = np.where(anc_cls[:,3]<origin_size[1], anc_cls[:,3], origin_size[1])
        anchors.append(anc_cls)
    
    return anchors
 

if __name__ == '__main__':
    while(1):
        picid = random.randint(0,24966)
        picid = '%05d'%picid + '.png'
        #picid = '14067.png'
        label = Image.open('/home/zhonghao/Documents/gen1/data/GTA5/labels/'+picid)
        h,w = 360, 640
        label = label.resize((w,h), Image.NEAREST)
        label = np.asarray(label, np.float32)
        id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                        26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
        label = torch.from_numpy(label_copy).long()
        t1 = time.time()
        #pdb.set_trace()
        anchors = anchorsbi(label)
        print(time.time()-t1, picid)
        save_img('/home/zhonghao/Documents/gen1/data/GTA5/images/'+picid, anchors)
        pdb.set_trace()
