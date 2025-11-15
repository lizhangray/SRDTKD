import logging
logger = logging.getLogger('base')


def get_HAT(checkpoint):
    from .hat_arch import get_HAT
    model = get_HAT(checkpoint=checkpoint)
    return model