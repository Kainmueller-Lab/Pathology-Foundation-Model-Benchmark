from mmpretrain.models.losses import CrossEntropyLoss

from mmengine.registry import MODELS

@MODELS.register_module()
class CELoss(CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #CrossEntropyLoss(use_sigmoid=False, use_soft=False, reduction='mean', loss_weight=1.0, class_weight=None, pos_weight=None)