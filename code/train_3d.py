import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la import (LAHeart, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case
from networks.vnet import VNet
from networks.ResNet34 import Resnet34

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=16,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
parser.add_argument('--data_name', type=str, default='LA', help='path to the data',choices=['LA'])
parser.add_argument('--item_id', type=str, default='', help='using item list training')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def power_law_scaling(tensor, power=2):
    min_val = tensor.min()
    max_val = tensor.max()
    return ((tensor - min_val) / (max_val - min_val)) ** power

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(name='vnet'):
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

    model1.train()
    model2.train()
    if args.data_name == 'LA':
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           train_flod='train.list',  # todo change training flod
                           common_transform=transforms.Compose([
                               RandomCrop(args.patch_size),
                               ToTensor(),
                           ]),
                           sp_transform=True)
    else:
        db_train = BraTS2019(base_dir=train_data_path,
                             split='train',
                             num=None,
                             transform=transforms.Compose([
                                 RandomRotFlip(),
                                 RandomCrop(args.patch_size),
                                 ToTensor(),
                             ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.data_name == 'LA':
        len_db_train = len(db_train)
        labeled_idxs = list(range(int(0.2 * len_db_train)))  # todo set labeled num
        unlabeled_idxs = list(range(int(0.2 * len_db_train), len_db_train))  # todo set labeled num all_sample_num
    else:
        labeled_idxs = list(range(0, args.labeled_num))
        unlabeled_idxs = list(range(args.labeled_num, 250))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    T=0.1
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.dice_loss
    consistency_criterion = losses.softmax_mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch1, label_batch = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch2 = sampled_batch[1]['image']
            volume_batch1, label_batch = volume_batch1.cuda(), label_batch.cuda()
            volume_batch2 = volume_batch2.cuda()

            outputs1 = model1(volume_batch1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch2)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))


            with torch.no_grad():
                weight = torch.abs(outputs_soft1[:args.labeled_bs, 1, :, :, :]-outputs_soft2[:args.labeled_bs, 1, :, :, :])
                weight = power_law_scaling(weight)


            v_mse_dist = consistency_criterion(outputs_soft1[:args.labeled_bs, 1, :, :, :], label_batch[:args.labeled_bs].float() )
            r_mse_dist = consistency_criterion(outputs_soft2[:args.labeled_bs, 1, :, :, :], label_batch[:args.labeled_bs].float())

            v_mse      = torch.sum(weight * v_mse_dist) / (torch.sum(weight) + 1e-16)
            r_mse      = torch.sum(weight * r_mse_dist) / (torch.sum(weight) + 1e-16)

            v_outputs_clone = outputs_soft1[args.labeled_bs:, :, :, :, :].clone().detach()
            r_outputs_clone = outputs_soft2[args.labeled_bs:, :, :, :, :].clone().detach()
            v_outputs_clone1 = torch.pow(v_outputs_clone, 1 / T)
            r_outputs_clone1 = torch.pow(r_outputs_clone, 1 / T)
            v_outputs_clone2 = torch.sum(v_outputs_clone1, dim=1, keepdim=True)
            r_outputs_clone2 = torch.sum(r_outputs_clone1, dim=1, keepdim=True)
            v_outputs_PLable = torch.div(v_outputs_clone1, v_outputs_clone2)
            r_outputs_PLable = torch.div(r_outputs_clone1, r_outputs_clone2)


            v_uncertainty_map = -1.0 * torch.sum(v_outputs_clone * torch.log(v_outputs_clone + 1e-6), dim=1,
                                                 keepdim=True)
            r_uncertainty_map = -1.0 * torch.sum(r_outputs_clone * torch.log(r_outputs_clone + 1e-6), dim=1,
                                                 keepdim=True)

            mask_v_lower = v_uncertainty_map < r_uncertainty_map

            Plabel = torch.where(mask_v_lower, v_outputs_PLable, r_outputs_PLable)

            r_consistency_dist = consistency_criterion(outputs_soft2[args.labeled_bs:, :, :, :, :], Plabel)
            b, c, w, h, d = r_consistency_dist.shape
            pseudo_supervision1 = torch.sum(r_consistency_dist) / (b * c * w * h * d)
            v_consistency_dist = consistency_criterion(outputs_soft1[args.labeled_bs:, :, :, :, :], Plabel)
            b, c, w, h, d = v_consistency_dist.shape
            pseudo_supervision2 = torch.sum(v_consistency_dist) / (b * c * w * h * d)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss +1.4*(v_mse+r_mse)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break

        if epoch_num > 0.8*max_epoch:
            save_mode_path = os.path.join(
                snapshot_path, 'model1_epoch_' + str(epoch_num) + '.pth')
            torch.save(model1.state_dict(), save_mode_path)
            logging.info("save model1 to {}".format(save_mode_path))

            save_mode_path = os.path.join(
                snapshot_path, 'model2_epoch_' + str(epoch_num) + '.pth')
            torch.save(model2.state_dict(), save_mode_path)
            logging.info("save model2 to {}".format(save_mode_path))
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # 建立训练时间戳
    Training_ID = time.strftime("%Y%m%d_%H%M_%S", time.localtime())
    args.item_id = str(Training_ID)

    snapshot_path = "../model/{}_{}/{}_{}".format(
        args.exp, args.labeled_num, args.model, args.item_id)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.info('+++++++++++++++Start Time+++++++++++++++')
    logging.info(str(Training_ID))

    train(args, snapshot_path)
