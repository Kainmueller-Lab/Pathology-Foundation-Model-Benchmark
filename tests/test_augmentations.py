import kornia.augmentation as Kaug
import torch

import benchmark.augmentations.custom_augmentations as Caug
from benchmark.augmentations.augmentations import Augmenter

params = {
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomVerticalFlip": {"p": 1.0},
}


def test_augmenter():
    augmenter = Augmenter(params=params)
    assert isinstance(augmenter, Augmenter)
    assert isinstance(augmenter.transforms[0], Kaug.RandomHorizontalFlip)
    assert isinstance(augmenter.transforms[1], Kaug.RandomVerticalFlip)


def test_augmenter_forward():
    # test augmenter forward pass
    augmenter = Augmenter(params=params)
    batch = torch.rand(2, 3, 32, 32)
    batch_aug = augmenter(batch)
    assert batch_aug.shape == batch.shape
    assert not torch.all(torch.eq(batch, batch_aug))
    # test augmenter with image and mask
    mask = torch.round(torch.rand(2, 3, 32, 32) * 10)
    augmenter = Augmenter(params=params, data_keys=["image", "mask"], same_on_batch=False)
    batch_aug, mask_aug = augmenter(batch, mask)
    assert batch_aug.shape == batch.shape
    assert mask_aug.shape == mask.shape
    assert not torch.all(torch.eq(batch, batch_aug))
    assert not torch.all(torch.eq(mask, mask_aug))
    for i in torch.unique(mask):
        sorted_batch, _ = torch.sort(batch[mask == i].flatten())
        sorted_batch_aug, _ = torch.sort(batch_aug[mask_aug == i].flatten())
        assert torch.all(torch.eq(sorted_batch, sorted_batch_aug))


def test_augmenter_inverse():
    augmenter = Augmenter(params=params, data_keys=["image", "mask"], same_on_batch=False)
    batch = torch.rand(2, 3, 32, 32)
    mask = torch.rand(2, 3, 32, 32) * 10
    batch_aug, mask_aug = augmenter(batch, mask)
    batch_inv, mask_inv = augmenter.inverse(batch_aug, mask_aug)
    assert batch_inv.shape == batch.shape
    assert mask_inv.shape == mask.shape
    assert torch.all(torch.eq(batch, batch_inv))
    assert torch.all(torch.eq(mask, mask_inv))


def test_repeat_last_transform():
    params["RandomRotation"] = {"p": 1.0, "degrees": [0, 360]}
    augmenter = Augmenter(params=params, data_keys=["image", "mask"], same_on_batch=False)
    batch = torch.rand(2, 3, 32, 32)
    mask = torch.rand(2, 3, 32, 32) * 10
    batch_aug_1, mask_aug_1 = augmenter(batch, mask)
    batch_aug_repeat, mask_aug_repeat = augmenter.repeat_last_transform(batch, mask)
    batch_aug_2, mask_aug_2 = augmenter(batch, mask)
    assert torch.all(torch.eq(batch_aug_1, batch_aug_repeat))
    assert torch.all(torch.eq(mask_aug_1, mask_aug_repeat))
    assert not torch.all(torch.eq(batch_aug_1, batch_aug_2))
    assert not torch.all(torch.eq(mask_aug_1, mask_aug_2))
