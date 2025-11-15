# -*- coding: utf-8 -*-
# @Time    : 07/27/2025 16:00 PM
# @Author  : Wedream
# @FileName: main.py
# @Software: PyCharm
# @Description: Training and Testing the Student

from Trainer_for_infer import Trainer
import argparse

parser = argparse.ArgumentParser(description="Initial the trainer setting.")

# the model would to be trained or tested.
parser.add_argument("--model_name", type=str, default="LBNet")
parser.add_argument("--checkpoint", type=str, default="DTKD-LBNet.pth")

# teacher_1 model with high score of psnr and ssim.
# teacher_2 model with high score of PI or else.
# if you want to train by the dataset instead of the teacher's result,
# you should set the --train_by_TT False in your shell command.
parser.add_argument("--train_by_TT",             default=True)
parser.add_argument("--train_by_TTT",            default=False,     action="store_true")
parser.add_argument("--t1_model_name", type=str, default="IMDN")
parser.add_argument("--t1_checkpoint", type=str, default="IMDN_x4.pth")
parser.add_argument("--t2_model_name", type=str, default="EdgeSRN", help='lpts')
parser.add_argument("--t2_checkpoint", type=str, default="EdgeSRN_x4.pth")

# Train setting
parser.add_argument("--Train",                  default=True)

# Test Setting
# you should change the argument --Train False in your shell command.
parser.add_argument("--test_folder", type=str,   default='Datasets2023/GTmod12_LRx4/Set5_LRbicx4')
parser.add_argument("--if_save_image",              default=True)
parser.add_argument("--output_folder", type=str,    default='./Result')

# Use default is enough
parser.add_argument("--seed",   type=int, default=2025)

# dataset root. annotate by wedream.
parser.add_argument("--root",   type=str, default="srdata",   help='dataset directory')
parser.add_argument("--scale",  type=int, default=4,                    help="super-resolution scale")
parser.add_argument("--phase",  type=str, default='train')

parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--n_val",      type=int, default=1, help="number of validation set")

# image setting
parser.add_argument("--isY",                    default=True,   action="store_true")
parser.add_argument("--ext",        type=str,   default='.png', help='png or npy')
parser.add_argument("--n_colors",   type=int,   default=3,      help="number of color channels to use")
parser.add_argument("--rgb_range",  type=int,   default=1,      help="maxium value of RGB")
parser.add_argument("--patch_size", type=int,   default=192,    help="output patch size")

args = parser.parse_args()

def display_args(args):
    print("#######################################################################################################")
    print(args)

    print("#######################################################################################################")

    print("===> Model_name         : " + str(args.model_name))
    print("===> Model_checkpoint   : " + str(args.checkpoint))

    if args.Train == True:
        print("===> Learn_rate_init    : " + str(args.lr))
        print("===> Rate_L1_loss       : " + str(args.Rate_l1))
        print("===> Rate_Lm_loss       : " + str(args.Rate_lm))
        print("===> Rate_Lpts_loss     : " + str(args.Rate_lpts))
        print("===> Rate_Lbp_loss      : " + str(args.Rate_lbp))
        print("===> Rate_Lfrac_loss    : " + str(args.Rate_lfrac))
        print("===> Batch_size         : " + str(args.batch_size))
        print("===> FlipRGB            : " + str(args.flipRGB))
        print("===> VGG_type           : " + str(args.VGG_type))
        print("===> Optimizer          : " + "adan" if args.adan == True else "adam")

        if args.train_by_TT == True:
            print("===> T1_Model_name      : " + str(args.t1_model_name))
            print("===> T1_Model_checkpoint: " + str(args.t1_checkpoint))
            print("===> T2_Model_name      : " + str(args.t2_model_name))
            print("===> T2_Model_checkpoint: " + str(args.t2_checkpoint))
            print("===> entropy_gate       : " + str(args.entropy_gate))
            print("===> loss_attn          : " + str(args.loss_attn))
            if args.resume:
                print("===> Resume             : " + str(args.resume))

            if args.train_by_TTT == True:
                print("===> T3_Model_name      : " + str(args.t3_model_name))
                print("===> T3_Model_checkpoint: " + str(args.t3_checkpoint))
    else:
        print("===> Test_datasets: " + str(args.test_datasets))

    print("#######################################################################################################")

if __name__ == '__main__':


    trainer = Trainer(args)
    if args.Train == True:
        display_args(args)
        trainer.Train()
    else:
        trainer.Test()

