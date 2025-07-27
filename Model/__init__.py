# -*- coding: utf-8 -*-
# @Time    : 7/15/2021 3:52 PM
# @Author  : YaoGengqi
# @FileName: __init__.py
# @Software: PyCharm
# @Description:

from .IMDN import get_IMDN
from .RFDN import get_RFDN
from .block import VGGFeatureExtractor as get_Extractor
from .EdgeSRN import get_EdgeSRN
from .network_swinir import get_SwinIR, get_SwinIR_GAN
from .LBNet import get_LBNet, get_LBNet_tiny
from .ESRGAN import get_ESRGAN

def get_model(model_name, checkpoint, upscale):

    if model_name == 'IMDN':
        return get_IMDN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'RFDN':
        return get_RFDN(upscale=upscale, checkpoint=checkpoint)

    elif model_name == 'EdgeSRN':
        return get_EdgeSRN(checkpoint=checkpoint)

    if model_name == 'SwinIR':
        return get_SwinIR(checkpoint=checkpoint)

    if model_name == 'SwinIR-GAN':
        return get_SwinIR_GAN(checkpoint=checkpoint)

    if model_name == 'LBNet':
        return get_LBNet(checkpoint=checkpoint)

    if model_name == 'LBNet_tiny':
        return get_LBNet_tiny(checkpoint=checkpoint)

    if model_name == 'ESRGAN':
        return get_ESRGAN(checkpoint=checkpoint)

    if model_name == 'SwinIRGAN':
        return get_SwinIR_GAN(checkpoint=checkpoint)