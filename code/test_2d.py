import argparse
import os
import shutil
import logging
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from torch.utils.data import DataLoader
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from dataloaders.dataset import (BaseDataSets, RandomGenerator,Synapse_dataset, RandomGenerator,
                                 TwoStreamBatchSampler)
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='isic', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
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
parser.add_argument('--patch_size', type=list,  default=[224, 224],
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
parser.add_argument('--labeled_num', type=int, default=207,
                    help='labeled data')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--data_name', type=str, default='kvasir', help='path to the data',choices=['isic2018', 'kvasir', 'CVC-ClinicDB','LA'])
parser.add_argument('--item_id', type=str, default='20240417_2331_09', help='using item list training')

FLAGS = parser.parse_args()
config = get_config(FLAGS)
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        # print(pred.shape,gt.shape)
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 1,0
    else:
        return 0, 0,0
    return dice, hd95, asd


# def test_single_volume(case, net1, net2,test_save_path, FLAGS):
def test_single_volume(image, label, case_name, net1, net2, test_save_path, FLAGS):
    # # print(FLAGS.root_path+ FLAGS.data_name+"/test_vol_h5/{}.npz".format(case))
    # data = np.load(FLAGS.root_path+ FLAGS.data_name+"/test_vol_h5/{}.npz".format(case))
    # image, label = data['image'], data['label']
    # image = torch.from_numpy(image.astype(np.float32))
    # image = image.permute(2, 0, 1)
    # # print(image.shape,label.shape)
    # prediction = np.zeros_like(label)
    # _, x, y = image.shape

    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _, x, y = image.shape
    # 缩放图像符合网络输入大小224x224
    if x != FLAGS.patch_size[0] or y != FLAGS.patch_size[1]:
        image = zoom(image, (1, FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()

    net1.eval()
    net2.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main1, _, _, _ = net1(input)
            out_main2, _, _, _ = net2(input)
        else:
            out_main1 = net1(input)
            out_main2 = net2(input)
        # print(torch.softmax(out_main1, dim=1))
        out_main=(out_main1+out_main2)/2
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != FLAGS.patch_size[0] or y != FLAGS.patch_size[1]:
            prediction = zoom(out, (x /  FLAGS.patch_size[0], y /  FLAGS.patch_size[1]), order=0)
        else:
            prediction = out

    first_metric = calculate_metric_percase(prediction==1, label==1)
    print(first_metric[0],case_name)

    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case_name + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case_name + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case_name + "_gt.nii.gz")

    # 将图像转换为ITK格式并设置空间间隔
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))

    # 将ITK图像转换为NumPy数组
    img_np = sitk.GetArrayFromImage(img_itk)
    prd_np = sitk.GetArrayFromImage(prd_itk)
    lab_np = sitk.GetArrayFromImage(lab_itk)
    def normalize_and_save(image_array, file_name):
        if image_array.shape[0]==3:
            # 首先调整数组的形状为(高度, 宽度, 通道数)
            image_array = np.transpose(image_array, (1, 2, 0))

        # 归一化图像数据到0-1
        image_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())

        # 转换为8位整型
        image_8bit = (255 * image_normalized).astype(np.uint8)

        # 使用PIL保存为JPEG
        Image.fromarray(image_8bit).save(file_name)

    # 保存为JPG格式（只取第一个切片作为示例）
    normalize_and_save(img_np,test_save_path + case_name + "_img.jpg")
    normalize_and_save(prd_np,test_save_path + case_name + "_pred.jpg")
    normalize_and_save(lab_np,test_save_path + case_name + "_gt.jpg")
    return first_metric


def Inference(FLAGS):
    # with open(FLAGS.root_path+ FLAGS.data_name+'/test_vol.txt', 'r') as f:
    #     image_list = f.readlines()
    # image_list = sorted([item.replace('\n', '').split(".")[0]
    #                      for item in image_list])
    db_test = Synapse_dataset(base_dir=FLAGS.root_path + FLAGS.data_name + '/test_vol_h5', list_dir=FLAGS.root_path + FLAGS.data_name,
                    split="test_vol",patch_size=FLAGS.patch_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))


    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=FLAGS.model, in_chns=3,
                            class_num=FLAGS.num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    net1 = create_model()
    # net2 = create_model()
    net2 = ViT_seg(config, img_size=FLAGS.patch_size,
                     num_classes=FLAGS.num_classes).cuda()
    max_metric = 0
    max_num =0

    # for num in range(int(546 * 0.8) + 2, 546):
    # for num in range(146, 182):
    # for num in range(301, 374):
    # for num in range(146, 181):
    for num in range(961, 1199):
        snapshot_path = "../model/{}_{}/{}_{}".format(
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id)
        test_save_path = "../model/{}_{}/{}_{}_predictions/{}.pth/".format(
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.item_id,num)
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)

        save_mode_path1 = os.path.join(snapshot_path, 'model1_epoch_' + str(num) + '.pth')
        save_mode_path2 = os.path.join(snapshot_path, 'model2_epoch_' + str(num) + '.pth')
        net1.load_state_dict(torch.load(save_mode_path1))
        net2.load_state_dict(torch.load(save_mode_path2))
        print("init weight from {}".format(save_mode_path1))
        print("init weight from {}".format(save_mode_path2))
        net1.eval()
        net2.eval()

        first_total = 0.0
        # for case in image_list:
        for i_batch, sampled_batch in enumerate(testloader):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            first_metric, second_metric, third_metric = test_single_volume(
                    image, label, case_name, net1, net2,test_save_path, FLAGS)

            # first_metric,second_metric,third_metric = test_single_volume(
            #     case, net1, net2,test_save_path, FLAGS)
            first_total += np.asarray(first_metric)
        avg_metric = first_total / len(db_test)
        print("num:{},avg metric: {}".format(num,avg_metric))
        if(avg_metric > max_metric):
            max_metric  = avg_metric
            max_num=num
        print("max_num:{},max metric: {}".format(max_num, max_metric))
    return max_metric,max_num


if __name__ == '__main__':

    metric,max_num = Inference(FLAGS)
    print("best:{} in {}".format(metric,max_num))
    # print((metric[0]+metric[1]+metric[2])/3)
