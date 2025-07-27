import logging
logger = logging.getLogger('base')


def get_LBNet(checkpoint):
    from .lbnet import get_LBNet
    model = get_LBNet(checkpoint=checkpoint)
    return model

def get_LBNet_tiny(checkpoint):
    from .lbnet_tiny import get_LBNet_tiny
    model = get_LBNet_tiny(checkpoint=checkpoint)
    return model
