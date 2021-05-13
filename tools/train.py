from numpy.lib.utils import source
import torch
import _init_paths
import argparse
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.metric import Metrics, setup_seed, test
from utils.metric import evaluate
from utils.loss import BceDiceLoss, BCELoss, DiceLoss, MaxSquareloss
from utils.polyp_dataset import Polyp
from models.mpa import MPA_model
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

BATCH_SIZE = 4
NUM_WORKERS = 4
POWER = 0.9
INPUT_SIZE = (256, 256)
SOURCE_DATA = '/home/cyang/MPA-DA/data/EndoScene'
TRAIN_SOURCE_LIST = '/home/cyang/MPA-DA/dataset/EndoScene_list/train.lst'
TEST_SOURCE_LIST = '/home/cyang/MPA-DA/dataset/EndoScene_list/test.lst'

TARGET_DATA = '/home/cyang/MPA-DA/data/Etislarib'
TRAIN_TARGET_LIST = '/home/cyang/MPA-DA/dataset/Etislarib_list/train_fold4.lst'
TEST_TARGET_LIST = '/home/cyang/MPA-DA/dataset/Etislarib_list/test_fold4.lst'

LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 1
NUM_STEPS = 150
VALID_STEPS = 100
GPU = '1'
FOLD = 'fold1'
TARGET_MODE = 'gt'
RESTORE_FROM = '/home/cyang/MPA-DA/checkpoint/Endo_best.pth'
SNAPSHOT_DIR = '/home/cyang/MPA-DA/checkpoint/'
SAVE_RESULT = False
RANDOM_MIRROR = True
IS_ADABN = False
IS_PSEUDO = False


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, length, power=0.9):
    lr = lr_poly(args.learning_rate, i_iter, NUM_STEPS * length, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=SOURCE_DATA,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--source-train", type=str, default=TRAIN_SOURCE_LIST,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--source-test", type=str, default=TEST_SOURCE_LIST,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--is_adabn", type=bool, default=IS_ADABN,
                        help="Whether to apply test mean and var rather than running.")
    parser.add_argument("--data-dir-target", type=str, default=TARGET_DATA,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--target-train", type=str, default=TRAIN_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--target-test", type=str, default=TEST_TARGET_LIST,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--valid-steps", type=int, default=VALID_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", type=bool, default=RANDOM_MIRROR,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=SAVE_RESULT,
                        help="Whether to save the predictions.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--fold", type=str, default=FOLD,
                        help="choose gpu device.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--is-pseudo", type=bool, default=IS_PSEUDO,
                        help="use pseudo labels.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_source = Polyp(root=args.data_dir, 
                        data_dir=args.source_train, mode='train', max_iter=None, is_mirror=args.random_mirror)
    test_source = Polyp(root=args.data_dir,
                        data_dir=args.source_test, mode='test', is_mirror=False)

    train_target = Polyp(root=args.data_dir_target,
                         data_dir=args.target_train, mode='train', max_iter=args.num_steps * train_source.__len__(), is_mirror=args.random_mirror)
    test_target = Polyp(root=args.data_dir_target,
                        data_dir=args.target_test, mode='test', is_mirror=False)

    train_loader = torch.utils.data.DataLoader(
        train_source, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_source, batch_size=1, shuffle=False, num_workers=args.num_workers)

    train_loader_target = torch.utils.data.DataLoader(
        train_target, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_target = torch.utils.data.DataLoader(
        test_target, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = MPA_model(1, pretrained=True).cuda()
    #model.load_state_dict(torch.load(args.restore_from))

    optimizer = torch.optim.SGD(
        model.optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    source_criterion = DiceLoss()
    #target_criterion = MaxSquareloss()
    target_batch = enumerate(train_loader_target)
    Best_dice = 0
    for epoch in range(args.num_steps):
        source_loss = 0
        target_loss = 0
        tic = time.time()
        model.train()
        for i_iter, batch in enumerate(train_loader):
            data, name = batch
            image = data['image']
            label = data['label']

            image = Variable(image).cuda()

            label = Variable(label).cuda()
            label = label.unsqueeze(1)

            _, batch_target = target_batch.__next__()
            data_target, name_target = batch_target
            image_target = data_target['image']
            label_target = data_target['label']
            image_target = Variable(image_target).cuda()

            coarse_s, fine_s, coarse_t, fine_t = model(image, image_target)

            loss = source_criterion(coarse_s, label) + source_criterion(fine_s, label) + source_criterion(coarse_t, label_target) + source_criterion(fine_t, label_target)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            source_loss += (source_criterion(coarse_s, label) + source_criterion(fine_s, label)).item()
            target_loss += (source_criterion(coarse_t, label_target) + source_criterion(fine_t, label_target)).item()
            lr = adjust_learning_rate(optimizer=optimizer, i_iter=i_iter + epoch * train_source.__len__(
            ) / args.batch_size, length=train_source.__len__() / args.batch_size)
            #lr = args.learning_rate
        batch_time = time.time() - tic
        print('Epoch: [{}/{}], Time: {:.2f}, '
              'lr: {:.6f}, Loss_source: {:.6f}, Loss_target: {:.6f}' .format(
                  epoch, args.num_steps, batch_time, lr, source_loss, target_loss))
        # begin test on target domain
        dice = test(model, test_loader_target, args)
        if Best_dice <= dice:
            Best_dice = dice
            torch.save(model.state_dict(), '/home/cyang/MPA-DA/checkpoint/MPA_best.pth')

if __name__ == '__main__':
    main()
