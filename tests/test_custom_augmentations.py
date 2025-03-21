import torch

from benchmark.augmentations.custom_augmentations import HEDNormalize


def test_hed_normalize():
    # Test instance creation
    hed_normalize = HEDNormalize(sigma=0.05, bias=0.1)
    assert isinstance(hed_normalize, HEDNormalize)
    assert hasattr(hed_normalize, "sigma")
    assert hasattr(hed_normalize, "bias")
    assert hasattr(hed_normalize, "rgb_t")
    assert hasattr(hed_normalize, "hed_t")

    batch_size = 4

    # Test forward pass
    img_batch = torch.rand(batch_size, 3, 32, 32)
    img_batch_aug = hed_normalize(img_batch)
    assert img_batch_aug.shape == img_batch.shape
    assert not torch.all(torch.eq(img_batch, img_batch_aug))

    # Test differentiability
    img_batch.requires_grad_(True)
    batch_aug_diff = hed_normalize(img_batch)
    loss = batch_aug_diff.sum()
    loss.backward()
    assert img_batch.grad is not None

    # Test repeatability
    batch_aug_1 = hed_normalize(img_batch)
    batch_aug_2 = hed_normalize(img_batch)
    assert not torch.allclose(batch_aug_1, batch_aug_2)  # Should be stochastic
