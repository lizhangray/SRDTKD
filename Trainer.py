# -*- coding: utf-8 -*-
# @Time    : 07/27/2025 16:00 PM
# @Author  : Wedream
# @FileName: Trainer.py
# @Software: PyCharm
# @Description: Training and Testing the model

import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from Datasets import DIV2K, Vaildation
from Utils import utils
import skimage.color as sc
import random
from Model import *
import datetime


class Trainer():

    def __init__(self, args):

        self.args = args
        self._setting()

        # 有checkpoint则加载，否则初始化
        self.model = get_model(model_name=args.model_name,
                               upscale=args.scale,
                               checkpoint=args.checkpoint).to(self.device)

        if self.args.Train == True:
            self._get_log()
            self._get_dataloader()


    # 测试时，一定要使用model.eval()
    def Test(self):

        dir_path = self.args.test_folder
        img_name = os.listdir(dir_path)
        img_path = [ os.path.join(dir_path, x) for x in img_name]

        for imname in img_path:

            # img operation
            im_l = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
            im_input = im_l / 255.0
            im_input = np.transpose(im_input, (2, 0, 1))
            im_input = im_input[np.newaxis, ...]
            im_input = torch.from_numpy(im_input).float()

            if self.cuda:
                self.model = self.model.to(self.device)
                im_input = im_input.to(self.device)

            # 测试时一定要加这个
            self.model.eval()

            with torch.no_grad():
                # print("im_input : {} in inference".format(im_input.size()))
                out = self.model(im_input)

            out_img = utils.tensor2np(out.detach()[0])

            # 存储输出图片
            if self.args.if_save_image:

                output_path = os.path.join(self.args.output_folder, self.args.checkpoint.split('.')[0], os.path.basename(self.args.test_folder))

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                output_folder = os.path.join(output_path, os.path.basename(imname))
                cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
                print("img_name: ===>  " + os.path.basename(imname) + "!")

        print("test_folder: ===>  " + dir_path + "!")
        print("save_folder: ===>  " + output_path + "!")


    # 训练批次进行验证并输出指标数据，保存日志。
    def _valid(self):

        self.model.eval()
        avg_psnr, avg_ssim = 0, 0
        for i, batch in enumerate(self.test_loader, 1):
            lr_tensor, hr_tensor = batch[0].to(self.device), batch[1].to(self.device)

            with torch.no_grad():
                pre = self.model(lr_tensor)

            sr_img = utils.tensor2np(pre.detach()[0])
            gt_img = utils.tensor2np(hr_tensor.detach()[0])

            crop_size = self.args.scale
            cropped_sr_img = utils.shave(sr_img, crop_size)
            cropped_gt_img = utils.shave(gt_img, crop_size)

            if self.args.isY is True:
                im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
                im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
            else:
                im_label = cropped_gt_img
                im_pre = cropped_sr_img

            avg_psnr += utils.compute_psnr(im_pre, im_label)
            avg_ssim += utils.compute_ssim(im_pre, im_label)

        avg_psnr = avg_psnr / len(self.test_loader)
        avg_ssim = avg_ssim / len(self.test_loader)

        if avg_psnr >= self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_ssim = avg_ssim
            self._save_checkpoint()

        # save latest weights every epoch
        cur_time = self.cur_time
        state_folder = "training_state/" + self.args.model_name + "/" + cur_time + "/"
        state_latest_path = state_folder + 'latest.pth'

        # 创建文件夹
        if not os.path.exists(state_folder):
            os.makedirs(state_folder)
        # 保存resume文件
        torch.save({
            'model': self.model.state_dict(),
            'lr': self.lr,
            'epoch': self.epoch,
            'lr_step_size': self.args.lr_step_size,
            'lr_gamma': self.args.lr_gamma,
            'best_ssim': self.best_ssim,
            'best_psnr': self.best_psnr

        }, state_latest_path)

        msg = "#### Valid({}): Epoch[{:03d}/{}]   ".format(self.args.valid_dataset, self.epoch, self.args.nEpochs)
        msg += "PSNR/Best_PSNR: {:.4f}/{:.4f}   SSIM/Best_SSIM: {:.4f}/{:.4f}   ".format(avg_psnr, self.best_psnr,
                                                                                         avg_ssim, self.best_ssim)
        msg += "LR_init: {:.7f}   ".format(self.args.lr)

        print(
            "\r                                                                                                                                  ",
            end="")
        print("\r" + msg, end="")
        with open(self.log_path, 'a') as self.log:
            print(str(datetime.datetime.now())[:19] + " INFO: " + msg[5:], file=self.log)

    # 存储权值文件
    def _save_checkpoint(self):

        args = self.args

        model_train_set = "_x" + str(args.scale) + "_"

        if self.train_by_TT == True:
            model_train_set += "t1_" + args.t1_model_name + "_t2_" + args.t2_model_name + "_"

        if args.Rate_l1 == 1:
            model_train_set += "L1_"
        else:
            model_train_set += str(args.Rate_l1) + "L1_"
        if args.Rate_lpts > 0:
            model_train_set += str(args.Rate_lpts) + "Lpts_"
        if args.Rate_lm > 0:
            model_train_set += str(args.Rate_lm) + "Lm_"


        # get time now
        cur_time = self.cur_time

        model_folder = "Checkpoints/" + args.model_name + "/" + cur_time + "/"

        model_out_path = model_folder + \
                         args.model_name + \
                         model_train_set + \
                         "{}_psnr_{:.3f}_ssim_{:.3f}_epoch_{}.pth".format(self.args.valid_dataset, self.best_psnr,
                                                                 self.best_ssim, self.epoch)

        # 如果存在文件夹，里面有之前指标较低的文件，则删除（节省空间）
        if os.path.exists(model_folder):

            files = glob.glob(model_folder + '*.pth')

            for f in files:
                try:
                    # print("Checkpoint {} remove!".format(f))
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
        # 否则创建文件夹
        else:
            os.makedirs(model_folder)

        torch.save(self.model.state_dict(), model_out_path)

        print("\r                                                                                                    ",
              end="")
        print("\r#### Checkpoint save in : " + model_out_path)

        with open(self.log_path, 'a') as self.log:
            print(str(datetime.datetime.now())[:19] + " INFO: Checkpoint save in : " + model_out_path, file=self.log)

    # 初始化设置
    def _setting(self):

        # set any seed
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # get the time when starting training
        currentDateAndTime = datetime.datetime.now()
        self.cur_time = "{}-{:02d}-{:02d}-{:02d}-{:02d}".format(currentDateAndTime.year, currentDateAndTime.month,
                                                                currentDateAndTime.day
                                                                , currentDateAndTime.hour, currentDateAndTime.minute)

        self.cuda = torch.cuda.is_available()

        if self.cuda:
            torch.backends.cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # self.epoch = self.args.start_epoch

        if self.args.Train == True:

            if self.args.valid_dataset == "Set5":
                # self.best_psnr = 30.2
                self.best_psnr = 20
                self.best_ssim = 0
            elif self.args.valid_dataset == "Set14":
                # self.best_psnr = 28
                self.best_psnr = 20
                self.best_ssim = 0
            else:
                # self.best_psnr = 20
                self.best_psnr = 10
                self.best_ssim = 0

            self.train_by_TT = self.args.train_by_TT

    # 用于获取数据集
    def _get_dataloader(self):

        train_dataset = DIV2K.div2k(self.args)

        hr_dir = 'Datasets2023/GTmod12/' + self.args.valid_dataset + '_GTmod12/'
        lr_dir = 'Datasets2023/GTmod12_LRx4/' + self.args.valid_dataset + '_LRbicx' + str(self.args.scale) + '/'

        valid_dataset = Vaildation.DatasetFromFolderVal(hr_dir, lr_dir, self.args.scale)

        # load training sets. annotate by wedream.
        self.train_loader = DataLoader(dataset=train_dataset,
                                       num_workers=self.args.num_workers,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=valid_dataset,
                                      num_workers=self.args.num_workers,
                                      batch_size=1,
                                      shuffle=False)

    # 用于计算训练所需的时长
    def _time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        print("Epoch_Time:%02d:%02d:%02d   " % (h, m, s), end="")

        seconds = seconds * (self.args.nEpochs - self.epoch)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 99:
            print("Rest_Time:%03d:%02d:%02d" % (h, m, s))
        else:
            print("Rest_Time:%02d:%02d:%02d" % (h, m, s))

    # 创建用于保存记录的log文件，并保存验证机的指标记录和checkpoint的保存记录。
    def _get_log(self):

        args = self.args

        # get time now
        cur_time = self.cur_time + ' '

        log_name = cur_time + args.model_name

        model_train_set = "_x" + str(args.scale) + "_"

        if self.train_by_TT == True:
            model_train_set += "t1_" + args.t1_model_name + "_t2_" + args.t2_model_name + "_"

        if args.Rate_l1 == 1:
            model_train_set += "L1_"
        else:
            model_train_set += str(args.Rate_l1) + "L1_"
        if args.Rate_lpts > 0:
            model_train_set += str(args.Rate_lpts) + "Lp_"
        if args.Rate_lm > 0:
            model_train_set += str(args.Rate_lm) + "Lm"

        log_name += model_train_set + ".txt"

        self.log_path = "logs/" + log_name
        with open(self.log_path, 'w') as self.log:
            for key in list(vars(args).keys()):
                print("Train_Args[%15s]:\t %s" % (key, vars(args)[key]), file=self.log)

