def compute_dice(pred, target, smooth=1e-5):
    """
    Compute Dice coefficient for binary segmentation.

    Args:
        pred (torch.Tensor): Predicted masks (N, H, W).
        target (torch.Tensor): Ground truth masks (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice
