import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import PIL.Image as Image
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
from model.deeplab_multi import DeeplabMultiFeature
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from utils.functions import *
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import synthiaDataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from skimage.measure import label as sklabel
from compute_iou import compute_mIoU
import pdb

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BG_LABEL = [0,1,2,3,4,8,9,10]
FG_LABEL = [5,6,7,11,12,13,14,15,16,17,18]

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/gta5_deeplab'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LIST_PATH_TARGET_TEST = './dataset/cityscapes_list/val.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 200000  # early stopping
NUM_PROTOTYPE = 50
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots_ssl/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'
SAVE_PATH = './result/cityscapes'
SSL_TARGET_DIR = './target_ssl_gt'

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_TARGET = 0.001

TARGET = 'cityscapes'
SET = 'train'

LAMBDA_ADV_CLS = 0.001
LAMBDA_ADV_INS = 0.001

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--ssl-target-dir", type=str, default=SSL_TARGET_DIR,
                        help="Path to folder storing the ground truth of the target dataset.")
    parser.add_argument("--data-list-target-test", type=str, default=DATA_LIST_PATH_TARGET_TEST,
                        help="Path to the file listing the images in the target val dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-cls", type=float, default=LAMBDA_ADV_CLS,
                        help="lambda_cls for adversarial training.")
    parser.add_argument("--lambda-adv-ins", type=float, default=LAMBDA_ADV_INS,
                        help="lambda_ins for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--num-prototype", type=int, default=NUM_PROTOTYPE,
                        help="Number of prototypes.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--continue-train", action="store_true",
                        help="continue training")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)

def seg_label(label):
    segs = []
    for fg in FG_LABEL:
        mask = label==fg
        if torch.sum(mask)>0:
            masknp = mask.cpu().numpy().astype(int)
            seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
            seg = torch.LongTensor(seg).cuda()
            pixelnum = np.zeros(forenum, dtype=int)
            for i in range(forenum):
                pixelnum[i] = torch.sum(seg==(i+1)).item()
            segs.append([seg, pixelnum])
        else:
            segs.append([mask.long(), np.zeros(0)])
    return segs



def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")
    cudnn.benchmark = True
    cudnn.enabled = True

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    Iter = 0
    bestIoU = 0

    # Create network
    # init G
    if args.model == 'DeepLab':
        model = DeeplabMultiFeature(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        if args.continue_train:
            if list(saved_state_dict.keys())[0].split('.')[0] == 'module':
                for key in saved_state_dict.keys():
                    saved_state_dict['.'.join(key.split('.')[1:])] = saved_state_dict.pop(key)
            model.load_state_dict(saved_state_dict)
        else:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes).to(device)

    if args.continue_train:
        model_weights_path = args.restore_from
        temp = model_weights_path.split('.')
        temp[-2] = temp[-2] + '_D'
        model_D_weights_path = '.'.join(temp)
        model_D.load_state_dict(torch.load(model_D_weights_path))
        temp = model_weights_path.split('.')
        temp = temp[-2][-9:]
        Iter = int(temp.split('_')[1]) + 1

    model.train()
    model.to(device)

    model_D.train()
    model_D.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # init data loader
    if args.data_dir.split('/')[-1] == 'gta5_deeplab':
        trainset = GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    elif args.data_dir.split('/')[-1] == 'syn_deeplab':
        trainset = synthiaDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    trainloader = data.DataLoader(trainset,
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set, ssl_dir=args.ssl_target_dir),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)

    # init optimizer
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", 
        keep_batchnorm_fp32=True, loss_scale="dynamic"
    )

    model_D, optimizer_D = amp.initialize(
        model_D, optimizer_D, opt_level="O2", 
        keep_batchnorm_fp32=True, loss_scale="dynamic"
    )

    # init loss
    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    L1_loss = torch.nn.L1Loss(reduction='none')

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    test_interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # init prototype
    num_prototype = args.num_prototype
    num_ins = args.num_prototype * 10
    src_cls_features = torch.zeros([len(BG_LABEL),num_prototype,2048], dtype=torch.float32).to(device)
    src_cls_ptr = np.zeros(len(BG_LABEL), dtype=np.uint64)
    src_ins_features = torch.zeros([len(FG_LABEL),num_ins,2048], dtype=torch.float32).to(device)
    src_ins_ptr = np.zeros(len(FG_LABEL), dtype=np.uint64)


    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    # start training
    for i_iter in range(Iter, args.num_steps):

        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0
        loss_cls_value = 0
        loss_ins_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D

            for param in model_D.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)

            src_feature, pred = model(images)
            pred_softmax = F.softmax(pred, dim=1)
            pred_idx = torch.argmax(pred_softmax, dim=1)

            right_label = F.interpolate(labels.unsqueeze(0).float(), (pred_idx.size(1),pred_idx.size(2)), mode='nearest').squeeze(0).long()
            right_label[right_label!=pred_idx] = 255

            for ii in range(len(BG_LABEL)):
                cls_idx = BG_LABEL[ii]
                mask = right_label==cls_idx
                if torch.sum(mask) == 0:
                    continue
                feature = global_avg_pool(src_feature, mask.float())
                if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.half()).float())).item():
                    continue
                src_cls_features[ii,int(src_cls_ptr[ii]%num_prototype),:] = torch.squeeze(feature).clone().detach()
                src_cls_ptr[ii] += 1

            seg_ins = seg_label(right_label.squeeze())
            for ii in range(len(FG_LABEL)):
                cls_idx = FG_LABEL[ii]
                segmask, pixelnum = seg_ins[ii]
                if len(pixelnum) == 0:
                    continue
                sortmax = np.argsort(pixelnum)[::-1]
                for i in range(min(10, len(sortmax))):
                    mask = segmask==(sortmax[i]+1)
                    feature = global_avg_pool(src_feature, mask.float())
                    if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.half()).float())).item():
                        continue
                    src_ins_features[ii,int(src_ins_ptr[ii]%num_ins),:] = torch.squeeze(feature).clone().detach()
                    src_ins_ptr[ii] += 1

            pred = interp(pred)
            loss_seg = seg_loss(pred, labels)
            loss = loss_seg

            # proper normalization
            loss = loss / args.iter_size
            amp_backward(loss, optimizer)
            loss_seg_value += loss_seg.item() / args.iter_size

            # train with target

            _, batch = targetloader_iter.__next__()
            images, trg_labels, _, _ = batch
            images = images.to(device)
            trg_labels = trg_labels.long().to(device)

            trg_feature, pred_target = model(images)

            pred_target_softmax = F.softmax(pred_target, dim=1)
            pred_target_idx = torch.argmax(pred_target_softmax, dim=1)

            loss_cls = torch.zeros(1).to(device)
            loss_ins = torch.zeros(1).to(device)
            if i_iter > 0:
                for ii in range(len(BG_LABEL)):
                    cls_idx = BG_LABEL[ii]
                    if src_cls_ptr[ii] / num_prototype <= 1:
                        continue
                    mask = pred_target_idx==cls_idx
                    feature = global_avg_pool(trg_feature, mask.float())
                    if cls_idx != torch.argmax(torch.squeeze(model.layer6(feature.half()).float())).item():
                        continue
                    ext_feature = feature.squeeze().expand(num_prototype, 2048)
                    loss_cls += torch.min(torch.sum(L1_loss(ext_feature, src_cls_features[ii,:,:]),dim=1) / 2048.)

                seg_ins = seg_label(pred_target_idx.squeeze())
                for ii in range(len(FG_LABEL)):
                    cls_idx = FG_LABEL[ii]
                    if src_ins_ptr[ii] / num_ins <= 1:
                        continue
                    segmask, pixelnum = seg_ins[ii]
                    if len(pixelnum) == 0:
                        continue
                    sortmax = np.argsort(pixelnum)[::-1]
                    for i in range(min(10, len(sortmax))):
                        mask = segmask==(sortmax[i]+1)
                        feature = global_avg_pool(trg_feature, mask.float())
                        feature = feature.squeeze().expand(num_ins, 2048)
                        loss_ins += torch.min(torch.sum(L1_loss(feature, src_ins_features[ii,:,:]),dim=1) / 2048.) / min(10, len(sortmax))

            pred_target = interp_target(pred_target)

            D_out = model_D(F.softmax(pred_target, dim=1))
            loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            pred_target = interp_target(pred_target)
            loss_seg_trg = seg_loss(pred_target, trg_labels)

            loss = loss_seg_trg + args.lambda_adv_target * loss_adv_target + args.lambda_adv_cls * loss_cls + args.lambda_adv_ins * loss_ins
            loss = loss / args.iter_size
            amp_backward(loss, optimizer)
            loss_adv_target_value += loss_adv_target.item() / args.iter_size

            # train D

            # bring back requires_grad

            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred.detach()
            D_out = model_D(F.softmax(pred, dim=1))

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss_D = loss_D / args.iter_size / 2
            amp_backward(loss_D, optimizer_D)
            loss_D_value += loss_D.item()

            # train with target
            pred_target = pred_target.detach()
            D_out = model_D(F.softmax(pred_target, dim=1))

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
            loss_D = loss_D / args.iter_size / 2
            amp_backward(loss_D, optimizer_D)
            loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_adv_target': loss_adv_target_value,
                'loss_D': loss_D_value,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_seg_trg = {3:.3f}, loss_adv = {4:.3f} loss_D = {5:.3f} loss_cls = {6:.3f} loss_ins = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value, loss_seg_trg.item(), loss_adv_target_value, loss_D_value, loss_cls.item(), loss_ins.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            testloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target_test, 
                                                           crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set='val'),
                                         batch_size=1, shuffle=False, pin_memory=True)
            model.eval()
            for index, batch in enumerate(testloader):
                image, _, name = batch
                with torch.no_grad():
                    output1, output2 = model(Variable(image).to(device))
                output = test_interp(output2).cpu().data[0].numpy()
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output = Image.fromarray(output)
                name = name[0].split('/')[-1]
                output.save('%s/%s' % (args.save, name))
            mIoUs = compute_mIoU(osp.join(args.data_dir_target,'gtFine/val'), args.save, 'dataset/cityscapes_list')
            mIoU = round(np.nanmean(mIoUs) * 100, 2)
            if mIoU > bestIoU:
                bestIoU = mIoU
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'BestGTA5.pth'))
                torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'BestGTA5_D.pth'))
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
            model.train()

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
