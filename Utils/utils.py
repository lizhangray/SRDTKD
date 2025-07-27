import numpy as np
import cv2
import math
import os
import torch
try:
    from skimage.measure import compare_psnr, compare_ssim
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim

#  计算psnr值
def compute_psnr(im1, im2, crop_border=0):

    if crop_border > 0:
        im1 = shave(im1, crop_border)
        im2 = shave(im2, crop_border)

    return compare_psnr(im1, im2)


#  计算ssim值
def compute_ssim(im1, im2, crop_border=0):

    if crop_border > 0:
        im1 = shave(im1, crop_border)
        im2 = shave(im2, crop_border)

    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

#  用于裁剪图像的边缘，在图像的检测指标之前。
def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im


# # 输入测试文件夹的名字， 返回HR和LRw文件夹的相对路径和图片格式
# def get_folder(dataset, scale):
#     hr_folder = 'Datasets/' + dataset + '/'
#     lr_folder = 'Datasets/' + dataset + '_LR/x' + str(scale) + '/'
#     if dataset == 'Set5':
#         ext = '.bmp'
#     elif dataset == 'Set14':
#         ext = '.png'
#     else:
#         ext = '.png'
#     return hr_folder, lr_folder, ext

# 输入测试文件夹的名字， 返回HR和LRw文件夹的相对路径和图片格式
def get_folder(dataset, scale):
    # hr_folder = 'Datasets/' + dataset + '/'
    # lr_folder = 'Datasets/' + dataset + '_LR/x' + str(scale) + '/'
    hr_folder = 'Datasets2023/GTmod12/' + dataset + '_GTmod12/'
    lr_folder = 'Datasets2023/GTmod12_LRx4/' + dataset + '_LRbicx' + str(scale) + '/'
    if dataset == 'Set5':
        ext = '.png'
        # ext = '.bmp'
    elif dataset == 'Set14':
        ext = '.png'
    else:
        ext = '.png'
    return hr_folder, lr_folder, ext

# 输入测试文件夹的名字， 返回HR和LRw文件夹的相对路径和图片格式
def get_folder_g7(dataset, scale):
    hr_folder = 'Datasets/' + dataset + '/'
    lr_folder = 'Datasets/' + dataset + '_LR/x' + str(scale) + '/'
    if dataset == 'Set5':
        ext = '.bmp'
    elif dataset == 'Set14':
        ext = '.png'
    else:
        ext = '.png'
    return hr_folder, lr_folder, ext


# 获取path路径下以.ext文件格式保存的图片路径列表
def get_list(path, ext):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]

def modcrop(im, modulo):
    sz = im.shape
    h = np.int32(sz[0] / modulo) * modulo
    w = np.int32(sz[1] / modulo) * modulo
    ims = im[0:h, 0:w, ...]
    return ims


def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def convert2np(tensor):
    return tensor.cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()

def get_entropy_origin(tensor):
    img = tensor2np(tensor)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

def get_entropy(tensor):
    img = tensor2np(tensor)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ->gray
    size_img = img.size
    # val = 0
    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数

    # print("hist_cv.size: {}".format(hist_cv.shape))

    P = hist_cv / (size_img)  # 概率
    # print("P: {}".format(P))
    p_list = [float(p * np.log2(1 / p)) if p != 0 else 0 for p in P]
    # print(test_list)
    res = np.sum(p_list)

    return res

def get_entropy_batch(tensor, entropy_thred):
    (n, c, h, w) = tensor.shape # NCHW
    min_tensor, min_index = [], [] # entropy小于阈值的tensor和对应下标
    max_tensor, max_index = [], [] # entropy大于阈值的tensor和对应下标
    for _i in range(n):
        img = tensor2np(tensor[_i]) # CHW->HWC
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ->gray
        size_img = img.size
        hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数
        P = hist_cv / (size_img)  # 概率
        p_list = [float(p * np.log2(1 / p)) if p != 0 else 0 for p in P]
        # print(test_list)
        res = np.sum(p_list)
        # print("tensor[{}]=>{}".format(_i,res))
        if res <= entropy_thred:
            min_tensor.append(torch.unsqueeze(tensor[_i], dim=0))
            min_index.append(_i)
        else:
            max_tensor.append(torch.unsqueeze(tensor[_i], dim=0))
            max_index.append(_i)

    # print("min_index: {}".format(min_index))
    if len(min_tensor)!=0:
        # 将小于阈值的张量整合到一起
        min_tensor = torch.cat(min_tensor, 0)
    if len(max_tensor)!=0:
        # 将大于阈值的张量整合到一起
        max_tensor = torch.cat(max_tensor, 0)

    return min_tensor, min_index, max_tensor, max_index


def get_l1_attention_mask(mask, loss):
    one = torch.ones_like(mask)
    zero = torch.zeros_like(mask)
    #print(torch.where(mask > loss, one, zero))
    return torch.where(mask > loss, one, zero), torch.where(mask > loss, mask, zero)


def get_deviation_batch(img, std_thresh=0.025):
    """
    输入:
        images: Tensor，形状为 (B, C, H, W)
        threshold: float，标准差阈值
    输出:
        below_thresh_indices: List[int]，标准差小于阈值的图像索引
        above_eq_thresh_indices: List[int]，标准差大于等于阈值的图像索引
    """
    # 计算每张图像的标准差，按(C, H, W)维度计算
    stds = img.view(img.size(0), -1).std(dim=1)

    # 找到小于阈值和大于等于阈值的下标
    min_index = (stds < std_thresh).nonzero(as_tuple=True)[0].tolist()
    max_index = (stds >= std_thresh).nonzero(as_tuple=True)[0].tolist()

    min_tensor = img[min_index]
    max_tensor = img[max_index]

    return min_tensor, min_index, max_tensor, max_index

import torch.nn.functional as F
def get_laplacian_batch(img, lap_thresh=0.025):
    """
    对输入图像应用拉普拉斯算子，返回边缘响应值是否小于阈值的下标。

    参数:
        images: Tensor，形状为 (B, C, H, W)
        threshold: float，边缘强度阈值

    返回:
        below_thresh_indices: List[int]，边缘强度小于阈值的图像索引
        above_eq_thresh_indices: List[int]，边缘强度大于等于阈值的图像索引
    """
    B, C, H, W = img.shape

    # 定义拉普拉斯核（适用于灰度图或每通道单独处理）
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=img.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)  # 形状: (1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)  # (C, 1, 3, 3) — 每个通道使用相同核

    # 对每个通道分别做卷积（组卷积）
    laplacian_response = F.conv2d(img, laplacian_kernel, padding=1, groups=C)  # shape: (B, C, H, W)

    # 计算每张图像的平均拉普拉斯响应（绝对值求平均，表示边缘强度）
    laplace_strength = laplacian_response.abs().view(B, C, -1).mean(dim=(1, 2))  # shape: (B,)

    # 判断是否小于阈值
    min_index = (laplace_strength < lap_thresh).nonzero(as_tuple=True)[0].tolist()
    max_index = (laplace_strength >= lap_thresh).nonzero(as_tuple=True)[0].tolist()

    min_tensor = img[min_index]
    max_tensor = img[max_index]

    return min_tensor, min_index, max_tensor, max_index


if __name__ == '__main__':
    input = torch.rand(8,3,256,256)
    input[4] = torch.zeros_like(input[4])
    input[5] = input[5]/50000
    min_tensor, min_index, max_tensor, max_index = get_laplacian_batch(input)
    print("out: {}".format((min_index, max_index)))