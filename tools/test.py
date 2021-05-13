import torch
import _init_paths
import argparse
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils.metric import Metrics, setup_seed, test
from utils.metric import evaluate
from utils.loss import BceDiceLoss, BCELoss, DiceLoss
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
TEST_SOURCE_LIST = '/home/cyang/MPA-DA/dataset/EndoScene_list/test.lst'

TARGET_DATA = '/home/cyang/MPA-DA/data/Etislarib'
TRAIN_TARGET_LIST = '/home/cyang/MPA-DA/dataset/Etislarib_list/train_fold4.lst'
TEST_TARGET_LIST = '/home/cyang/MPA-DA/dataset/Etislarib_list/test_fold4.lst'
ADABN_LIST = '/home/cyang/MPA-DA/dataset/Etislarib_list/adabn_fold4.lst'

NUM_CLASSES = 1
GPU = '7'
TRANSFER = True
TARGET_MODE = 'gt'
RESTORE_FROM = '/home/cyang/MPA-DA/checkpoint/EndoScene.pth'
SNAPSHOT_DIR = '/home/cyang/MPA-DA/checkpoint/'
SAVE_RESULT = True
RANDOM_MIRROR = False
IS_ADABN = True

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
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--random-mirror", type=bool, default=RANDOM_MIRROR,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-result", type=bool, default=SAVE_RESULT,
                        help="Whether to save the predictions.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()


def main():
    setup_seed(20)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    test_target = Polyp(root=args.data_dir_target,
                        data_dir=args.target_test, mode='test', is_mirror=args.random_mirror, is_inverse=args.is_inverse)

    test_loader_target = torch.utils.data.DataLoader(
        test_target, batch_size=1, shuffle=False, num_workers=args.num_workers)
    

    model = MPA_model(1, pretrained=False).cuda()
    model.load_state_dict(torch.load(args.restore_from))
    dice = test(model, test_loader_target, args, test_loader_target)

if __name__ == '__main__':
    main()
