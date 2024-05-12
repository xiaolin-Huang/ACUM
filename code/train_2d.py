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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,Synapse_dataset, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from collections import deque
from torchsummary import summary
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='kvasir', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
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

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=207,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--labeled_percentage', type=float, default=0.05, help='the percentage of labeled data')
parser.add_argument('--data_name', type=str, default='kvasir', help='the name of dataset')
parser.add_argument('--item_id', type=str, default='', help='using item list training')


args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
        return ref_dict[str(patiens_num)]
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
        return ref_dict[str(patiens_num)]
    else:
        print("Error")



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def power_law_scaling(tensor, power=2):
    min_val = tensor.min()
    max_val = tensor.max()
    return ((tensor - min_val) / (max_val - min_val)) ** power

def euclidean_distance(model_1, model_2):
    distance = 0.0
    for (param_1, param_2) in zip(model_1.parameters(), model_2.parameters()):
        distance += torch.norm(param_1 - param_2, 2).item()**2
    return distance**0.5

def cosine_similarity(model_1, model_2):

    dot_product = 0.0
    norm_1 = 0.0
    norm_2 = 0.0
    for (param_1, param_2) in zip(model_1.parameters(), model_2.parameters()):
        dot_product += torch.dot(param_1.view(-1), param_2.view(-1)).item()
        norm_1 += torch.norm(param_1, 2).item()**2
        norm_2 += torch.norm(param_2, 2).item()**2
    return dot_product / ((norm_1**0.5) * (norm_2**0.5))

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.data_name=="isic2018" or 'kvasir':
        print("!args.data_name",args.data_name)
        db_train = Synapse_dataset(base_dir=args.root_path+ args.data_name+'/train_npz', list_dir=args.root_path+ args.data_name, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(args.patch_size)]),patch_size=args.patch_size)
    else:
        db_train = BaseDataSets(base_dir=args.root_path + args.data_name, split="train", num=None, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
        db_val = BaseDataSets(base_dir=args.root_path + args.data_name, split="val")

    total_slices = len(db_train)

    if args.data_name=="isic2018" or 'kvasir':
        labeled_slice = int(total_slices * args.labeled_percentage)
    else:
        labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    consistency_criterion = losses.softmax_mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    entropy = []
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch1,volume_batch2, label_batch = sampled_batch[0]['image'], sampled_batch[1]['image'], sampled_batch[0]['label']
            volume_batch1,volume_batch2, label_batch = volume_batch1.cuda(), volume_batch2.cuda(), label_batch.cuda()

            outputs1 = model1(volume_batch1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)


            with torch.no_grad():
                weight = torch.abs(outputs_soft1[:args.labeled_bs, 1, :, :]-outputs_soft2[:args.labeled_bs, 1, :, :])
                weight = power_law_scaling(weight)

            pred1_mse_dist = consistency_criterion(outputs_soft1[:args.labeled_bs, 1, :, :], label_batch[:args.labeled_bs].squeeze(1).float())
            pred2_mse_dist = consistency_criterion(outputs_soft2[:args.labeled_bs, 1, :, :], label_batch[:args.labeled_bs].squeeze(1).float())

            loss1_mse = torch.sum(weight * pred1_mse_dist) / (torch.sum(weight) + 1e-16)
            loss2_mse = torch.sum(weight * pred2_mse_dist) / (torch.sum(weight) + 1e-16)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].squeeze(1).long()) + losses.dice_loss(
                outputs_soft1[:args.labeled_bs, 1, :, :], label_batch[:args.labeled_bs].squeeze(1)==1))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].squeeze(1).long()) + losses.dice_loss(
                outputs_soft2[:args.labeled_bs, 1, :, :], label_batch[:args.labeled_bs].squeeze(1)==1))


            T = 0.1
            with torch.no_grad():
                pred1_u_feature1 = torch.pow(outputs_soft1[args.labeled_bs:], 1 / T)
                pred2_u_feature1 = torch.pow(outputs_soft2[args.labeled_bs:], 1 / T)
                pred1_u_feature2 = torch.sum(pred1_u_feature1, dim=1, keepdim=True)
                pred2_u_feature2 = torch.sum(pred2_u_feature1, dim=1, keepdim=True)
                pred1_outputs_PLable = torch.div(pred1_u_feature1, pred1_u_feature2)
                pred2_outputs_PLable = torch.div(pred2_u_feature1, pred2_u_feature2)

                pred1_uncertainty_map = -1.0 * torch.sum(outputs_soft1[args.labeled_bs:] * torch.log(outputs_soft1[args.labeled_bs:] + 1e-6), dim=1,
                                                         keepdim=True)
                pred2_uncertainty_map = -1.0 * torch.sum(outputs_soft2[args.labeled_bs:] * torch.log(outputs_soft2[args.labeled_bs:] + 1e-6), dim=1,
                                                         keepdim=True)

                mask_v_lower = pred1_uncertainty_map < pred2_uncertainty_map
                Plabel = torch.where(mask_v_lower, pred1_outputs_PLable, pred2_outputs_PLable)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist1 = consistency_criterion(outputs_soft1[args.labeled_bs:], Plabel)
            consistency_dist2 = consistency_criterion(outputs_soft2[args.labeled_bs:], Plabel)
            b, c, w, h = consistency_dist1.shape

            pseudo_supervision1 = torch.sum(consistency_dist1) / (b * c * w * h)
            pseudo_supervision2 = torch.sum(consistency_dist2) / (b * c * w * h)


            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss

            loss = loss + 1.4*(loss1_mse+loss2_mse)

            pixel_entropies =-1.0 * torch.sum(Plabel * torch.log(Plabel + 1e-6), dim=1,
                             keepdim=True)
            average_entropy = torch.mean(pixel_entropies)

            entropy.append(average_entropy)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_


            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num >= max_iterations:
                break
            time1 = time.time()

        if epoch_num > 0:
            save_mode_path = os.path.join(
                snapshot_path, 'model1_epoch_' + str(epoch_num) + '.pth')
            torch.save(model1.state_dict(), save_mode_path)
            logging.info("save model1 to {}".format(save_mode_path))

            save_mode_path = os.path.join(
                snapshot_path, 'model2_epoch_' + str(epoch_num) + '.pth')
            torch.save(model2.state_dict(), save_mode_path)
            logging.info("save model2 to {}".format(save_mode_path))


        if iter_num >= max_iterations:
            iterator.close()
            break
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
    # 建立训练时间戳
    Training_ID = time.strftime("%Y%m%d_%H%M_%S", time.localtime())
    args.item_id = str(Training_ID)

    snapshot_path = "../model/{}_{}/{}_{}".format(
        args.exp, args.labeled_num, args.model,args.item_id)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    logging.info('+++++++++++++++Start Time+++++++++++++++')
    logging.info(str(Training_ID))

    train(args, snapshot_path)
