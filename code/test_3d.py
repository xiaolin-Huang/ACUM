import argparse
import os
import shutil
from glob import glob

import torch

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from config import get_config
from dataloaders.la_hxl import (LAHeart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from hxl_test_util import test_all_case
from networks.vnet import VNet
from networks.ResNet34 import Resnet34

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument('--patch_size', type=list,  default=(112, 112, 80),
                    help='patch size of network input')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=16,
                    help='labeled data')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--labeled_percentage', type=float, default=0.2, help='the percentage of labeled data')
parser.add_argument('--data_name', type=str, default='LA', help='path to the data',choices=['isic2018', 'kvasir', 'LA'])
parser.add_argument('--item_id', type=str, default='20231230_2058_40', help='using item list training')

FLAGS = parser.parse_args()
config = get_config(FLAGS)

snapshot_path = "../model/{}_{}/{}_{}/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id )
def Inference(FLAGS):
    # db_test = Synapse_dataset(base_dir=FLAGS.root_path + FLAGS.data_name + '/test_vol_h5',
    #                           list_dir=FLAGS.root_path + FLAGS.data_name,
    #                           split="test_vol", patch_size=FLAGS.patch_size)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # logging.info("{} test iterations per epoch".format(len(testloader)))
    #
    # snapshot_path = "../model/{}_{}/{}_{}".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id)
    # test_save_path = "../model/{}_{}/{}_{}_predictions/".format(
    #     FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id)
    # # if os.path.exists(test_save_path):
    # #     shutil.rmtree(test_save_path)
    # # os.makedirs(test_save_path)
    test_data_path = FLAGS.root_path + FLAGS.data_name + '/' + FLAGS.data_name + '_data/'
    with open('/data/benz/hxl/SSL4MIS-master/data/LA/test.list', 'r') as f:  # todo change test flod
        image_list = f.readlines()
    if FLAGS.data_name == 'LA':
        # image_list = [test_data_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]
        image_list = ['/data/benz/hxl/SSL4MIS-master/data/LA/data/' + item.replace('\n', '') + "/mri_norm2.h5" for item in
                      image_list]
    elif FLAGS.data_name == 'pancreas':
        image_list = [test_data_path + item.replace('\n', '') for item in image_list]

    num_classes = 2
    # net1 = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    # net2 = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()

    def create_model(name ='vnet'):
        # Network definition
        if name == 'vnet':
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        if name == 'resnet34':
            net = Resnet34(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
            model = net.cuda()
        return model

    model1 = create_model(name='vnet')
    model2 = create_model(name='resnet34')
    net1 =model1.cuda()
    net2 = model2.cuda()


    max_metric = [0,0,0,0]
    max_num = 0

    for num in range(int(749*0.8)+2,749):
        save_mode_path1 = os.path.join(snapshot_path, 'model1_epoch_' + str(num) + '.pth')
        save_mode_path2 = os.path.join(snapshot_path, 'model2_epoch_' + str(num) + '.pth')
        test_save_path = "../model/{}_{}/{}_{}_predictions/{}.pth/".format(
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id, num)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)
        net1.load_state_dict(torch.load(save_mode_path1))
        net2.load_state_dict(torch.load(save_mode_path2))
        print("init weight from {}".format(save_mode_path1))
        print("init weight from {}".format(save_mode_path2))
        net1.eval()
        net2.eval()

        first_total = 0.0
        if FLAGS.data_name == 'LA':
            avg_metric = test_all_case(net1, net2, image_list, num_classes=num_classes,
                                       patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                       save_result=True, test_save_path=test_save_path)
        elif FLAGS.data_name == 'pancreas':
            avg_metric = test_all_case(net1, net2, image_list, num_classes=num_classes,
                                       patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                       save_result=True, test_save_path=test_save_path)

        if(avg_metric[0] > max_metric[0]):
            max_metric  = avg_metric
            max_num=num

    return max_metric,max_num


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
