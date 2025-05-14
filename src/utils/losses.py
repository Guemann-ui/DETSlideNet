import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    # Remove the extra dimension
    mask = mask.squeeze(2)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # Compute binary cross entropy loss
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))  # Sum across spatial dimensions
    pred = torch.sigmoid(pred)

    # Compute intersection and union for IoU
    inter = ((pred * mask) * weit).sum(dim=(2, 3))  # Weighted intersection
    union = ((pred + mask) * weit).sum(dim=(2, 3))  # Weighted union

    # Compute weighted intersection-over-union (WIoU)
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # Return the mean of WBCE and WIoU
    return (wbce + wiou).mean()


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
