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
    batch = torch.rand(batch_size, 3, 32, 32)
    batch_aug = hed_normalize(batch)
    assert batch_aug.shape == batch.shape
    assert not torch.all(torch.eq(batch, batch_aug))

    # Test differentiability
    batch.requires_grad_(True)
    batch_aug_diff = hed_normalize(batch)
    loss = batch_aug_diff.sum()
    loss.backward()
    assert batch.grad is not None

    # Test compatibility with masks
    mask = torch.round(torch.rand(batch_size, 1, 32, 32))
    batch_aug_mask = hed_normalize(mask)
    assert batch_aug_mask.shape == batch.shape

    # Test repeatability
    batch_aug_1 = hed_normalize(batch)
    batch_aug_2 = hed_normalize(batch)
    assert not torch.allclose(batch_aug_1, batch_aug_2)  # Should be stochastic
