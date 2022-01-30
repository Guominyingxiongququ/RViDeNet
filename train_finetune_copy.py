from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from skimage.measure import compare_psnr,compare_ssim
from tensorboardX import SummaryWriter
from models import RViDeNet, RViDeNet_simple
from utils import *

parser = argparse.ArgumentParser(description='Finetune denoising model')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=70, help='num_epochs')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
args = parser.parse_args()

print(args.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

save_dir = '/disk1/xinyuan/result/finetune_simple_denoise'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

ps = args.patch_size  # patch size for training
batch_size = args.batch_size # batch size for training

log_dir = './logs/finetune_simple'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

denoiser = RViDeNet_simple().cuda()
initial_epoch = findLastCheckpoint(save_dir=save_dir)  
print('initial epoch: {}'.format(initial_epoch))
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    denoiser = torch.load(os.path.join(save_dir, 'model_epoch%d.pth' % initial_epoch))
    initial_epoch += 1
# else:
#     denoiser = torch.load(os.path.join('pretrain_model/model_epoch33.pth')).cuda()

recon_params1 = list(map(id, denoiser.recon_trunk.parameters()))
recon_params2 = list(map(id, denoiser.cbam.parameters()))
recon_params3 = list(map(id, denoiser.conv_last.parameters()))
base_params = filter(lambda p: id(p) not in recon_params1+recon_params2+recon_params3, denoiser.parameters())

opt = optim.Adam([{'params': base_params}, {'params': denoiser.recon_trunk.parameters(), 'lr': 1e-5}, {'params': denoiser.cbam.parameters(), 'lr': 1e-5}, {'params': denoiser.conv_last.parameters(), 'lr': 1e-5}], lr = 1e-6)

train_data_length = 6*5*7

iso_list = [1600,3200,6400,12800,25600]

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(train_data_length/batch_size)
temporal_frames_num = 3
for epoch in range(initial_epoch, args.num_epochs+1):
    cnt = 0
    for batch_id in range(int(train_data_length/batch_size)):
        input_batch_list = []
        gt_rgb_batch_list = []
        self_consistency_batch_list = []
        noisy_level_batch_list = []
        batch_num = 0
        while batch_num<batch_size:
            batch_num += 1
            scene_ind = np.random.randint(1,6+1)
            frame_ind = np.random.randint(2,6+1)
            noisy_level = np.random.randint(1,5+1)
            noisy_frame_index_for_current = np.random.randint(0,6+1)

            noisy_level_batch_list.append(np.expand_dims((noisy_level-1)*9/4, axis=0))

            input_pack_list = []
            gt_rgb_pack_list = []
            self_consistency_pack_list = []
            H = 1080
            W = 1920

            xx = np.random.randint(0, W - ps + 1)
            while xx%2!=0:
                xx = np.random.randint(0, W - ps + 1)
            yy = np.random.randint(0, H - ps + 1)
            while yy%2!=0:
                yy = np.random.randint(0, H - ps + 1)

            for shift in range(-1,2):
                
                gt_rgb_full = cv2_read('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.png'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift))
                gt_rgb_pack = gt_rgb_full[:, yy:yy + ps, xx:xx + ps, :]

                if shift == 0:
                    for self_consistency_index in range(4):
                        noisy_rgb = cv2_read('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_noisy/scene{}/ISO{}/frame{}_noisy{}.png'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_current+self_consistency_index))
                        noisy_rgb_full = noisy_rgb
                        noisy_pack = noisy_rgb_full[:, yy:yy + ps, xx:xx + ps, :]
                        self_consistency_pack_list.append(noisy_pack)
                    input_pack_list.append(self_consistency_pack_list[1])
                else:
                    noisy_frame_index_for_other = np.random.randint(0,9+1)
                    noisy_rgb = cv2_read('/disk1/xinyuan/data/CRVD_dataset/indoor_rgb_noisy/scene{}/ISO{}/frame{}_noisy{}.png'.format(scene_ind, iso_list[noisy_level-1], frame_ind+shift, noisy_frame_index_for_other))
                    noisy_rgb_full = noisy_rgb
                    noisy_pack = noisy_rgb_full[:, yy:yy + ps, xx:xx + ps, :]
                    input_pack_list.append(noisy_pack)
        
                gt_rgb_pack_list.append(gt_rgb_pack)
     
            self_consistency_pack_frames = np.concatenate(self_consistency_pack_list, axis=3)
            input_pack_frames = np.concatenate(input_pack_list, axis=3)
            gt_rgb_pack = gt_rgb_pack_list[1]

            input_batch_list.append(input_pack_frames)
            gt_rgb_batch_list.append(gt_rgb_pack)
            self_consistency_batch_list.append(self_consistency_pack_frames)

        input_batch = np.concatenate(input_batch_list, axis=0)
        gt_rgb_batch = np.concatenate(gt_rgb_batch_list, axis=0)
        self_consistency_batch = np.concatenate(self_consistency_batch_list, axis=0)
        noisy_level_batch = np.expand_dims(np.expand_dims(np.expand_dims(np.concatenate(noisy_level_batch_list, axis=0), axis=1), axis=2), axis=3)

        in_data = torch.from_numpy(input_batch.copy()).permute(0,3,1,2).cuda()
        gt_rgb_data = torch.from_numpy(gt_rgb_batch.copy()).permute(0,3,1,2).cuda()
        self_consistency_data = torch.from_numpy(self_consistency_batch.copy()).permute(0,3,1,2).cuda()
        noisy_level_data = torch.from_numpy(noisy_level_batch.copy()).float().cuda()
         
        denoiser.train()
        opt.zero_grad()

        denoised_out = denoiser(in_data.reshape(batch_size,3,3,ps,ps))

        denoised_out1 = denoiser(self_consistency_data.reshape(batch_size,4,3,ps,ps)[:,0:3,:,:,:])
        denoised_out2 = denoiser(self_consistency_data.reshape(batch_size,4,3,ps,ps)[:,1:4,:,:,:])

        #rgb l1 loss
        rgb_l1_loss = reduce_mean(denoised_out, gt_rgb_data) + reduce_mean(denoised_out1, gt_rgb_data)*0.1 + reduce_mean(denoised_out2, gt_rgb_data)*0.1

        # self consistency loss to solve flickering
        self_consistency_loss = reduce_mean_with_weight(denoised_out1, denoised_out2, noisy_level_data)
        
        loss = rgb_l1_loss + self_consistency_loss
        loss.backward()
        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('rgb_l1_loss', rgb_l1_loss.item(), step)
        writer.add_scalar('self_consistency_loss', self_consistency_loss.item(), step)
        print("epoch:%d iter%d loss=%.6f" % (epoch, cnt, loss.data))
 
    if epoch%1==0:
        torch.save(denoiser, os.path.join(save_dir, 'model_epoch%d.pth' % epoch))
