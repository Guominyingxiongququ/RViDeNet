from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from utils import *
from shutil import copyfile


parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--model', dest='model', type=str, default='finetune_rgb', help='model type')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--output_dir', type=str, default='/disk1/xinyuan/result/pretrain_denoise_result/', help='output path')
parser.add_argument('--vis_data', type=bool, default=True, help='whether to visualize noisy and gt data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# isp = torch.load('isp/ISP_CNN.pth').cuda()
# model = torch.load('model/pretrain.pth').cuda()
model_path = "/disk1/xinyuan/result/finetune_simple_denoise/model_epoch70.pth"

model = torch.load(model_path).cuda()

iso_list = [1600, 3200, 6400, 12800, 25600]
for iso in iso_list:
    print('processing iso={}'.format(iso))

    if not os.path.isdir(args.output_dir+'ISO{}'.format(iso)):
        os.makedirs(args.output_dir+'ISO{}'.format(iso))

    f = open('{}_model_test_psnr_and_ssim_on_iso{}.txt'.format(args.model, iso), 'w')

    context = 'ISO{}'.format(iso) + '\n'
    f.write(context)
  
    scene_avg_srgb_psnr = 0
    scene_avg_srgb_ssim = 0

    # for scene_id in range(1,5+1):
    for scene_id in range(8, 12):
        context = 'scene{}'.format(scene_id) + '\n'
        f.write(context)

        frame_avg_srgb_psnr = 0
        frame_avg_srgb_ssim = 0

        for i in range(1, 8):
            frame_list = []
            for j in range(-1, 2):
            # for j in [0, 0, 0]:
                if (i+j)<1:
                    lq= cv2.imread('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_noisy/scene{}/ISO{}/frame1_noisy0.png'.format(scene_id, iso),-1)
                    lq = lq[:, :, ::-1]
                    lq = lq.astype(np.float32)
                    lq = lq/255.0
                    frame_list.append(lq)
                elif (i+j)>7:
                    lq = cv2.imread('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_noisy/scene{}/ISO{}/frame7_noisy0.png'.format(scene_id, iso),-1)
                    lq = lq[:, :, ::-1]
                    lq = lq.astype(np.float32)
                    lq = lq/255.0
                    frame_list.append(lq)
                else:
                    lq = cv2.imread('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_noisy/scene{}/ISO{}/frame{}_noisy0.png'.format(scene_id, iso, i+j),-1)
                    lq = lq[:, :, ::-1]
                    lq = lq.astype(np.float32)
                    lq = lq/255.0
                    frame_list.append(lq)
            input_data = np.concatenate(frame_list, axis=2) 
            
            test_result = test_big_size_rgb(input_data, model, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64, video=True)
            test_result = test_result * 255
            test_result[test_result>255.0] = 255.0
            test_result[test_result<0.0] = 0.0
            test_result = np.uint8(test_result[0, :, :, ::-1])
            test_gt = cv2.imread('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.png'.format(scene_id, iso, i),-1)
            test_gt_float = test_gt.astype(np.float32)
            test_rgb_psnr = compare_psnr(test_gt_float, test_result, data_range=255.0)
            test_rgb_ssim = compare_ssim(test_gt_float, test_result, data_range=255.0, multichannel=True)
            print('scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(scene_id, i, test_rgb_psnr, test_rgb_ssim))
            context = 'rgb psnr/ssim: {}/{}'.format(test_rgb_psnr, test_rgb_ssim) + '\n'
            f.write(context)
            frame_avg_srgb_psnr += test_rgb_psnr
            frame_avg_srgb_ssim += test_rgb_ssim

            if args.vis_data:
                cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_noisy_sRGB.png'.format(iso, scene_id, i), test_result)

            if args.vis_data:
                cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_gt_sRGB.png'.format(iso, scene_id, i), test_gt)
            
            #output color map
            test_diff = np.abs(test_gt.astype(np.float32)/255 - test_result.astype(np.float32)/255)
            error_map = cv2.applyColorMap((test_diff*255).astype(np.uint8)*10, 2)
            cv2.imwrite(args.output_dir+'ISO{}/scene{}_frame{}_error_sRGB.png'.format(iso, scene_id, i), error_map)

            frame_avg_srgb_psnr += test_rgb_psnr
            frame_avg_srgb_ssim += test_rgb_ssim

        frame_avg_srgb_psnr = frame_avg_srgb_psnr/7
        frame_avg_srgb_ssim = frame_avg_srgb_ssim/7

        context = 'frame average srgb psnr:{},frame average srgb ssim:{}'.format(frame_avg_srgb_psnr,frame_avg_srgb_ssim) + '\n'
        f.write(context)

        scene_avg_srgb_psnr += frame_avg_srgb_psnr
        scene_avg_srgb_ssim += frame_avg_srgb_ssim

    scene_avg_srgb_psnr = scene_avg_srgb_psnr/4
    scene_avg_srgb_ssim = scene_avg_srgb_ssim/4
    context = 'scene average srgb psnr:{},scene frame average srgb ssim:{}'.format(scene_avg_srgb_psnr,scene_avg_srgb_ssim) + '\n'
    f.write(context)



