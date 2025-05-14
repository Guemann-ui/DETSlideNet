import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm


def compute_metrics(pred, gt):
    """
    Compute evaluation metrics: Precision, Recall, Dice, mIoU, HD95, OA.

    Args:
        pred (torch.Tensor): Binary prediction mask (B x 1 x H x W).
        gt   (torch.Tensor): Ground-truth mask     (B x 1 x H x W).

    Returns:
        tuple: precision, recall, dice, mIoU, hd95, OA
    """
    pred_flat = (pred.view(-1).cpu().numpy() > 0.5).astype(int)
    gt_flat = (gt.view(-1).cpu().numpy() > 0.5).astype(int)

    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)

    intersection = np.sum(pred_flat * gt_flat)
    dice = 2 * intersection / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-8)

    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    mIoU = intersection / (union + 1e-8)

    # HD95
    def hd95(a, b):
        idx_a = np.where(a == 1)[0]
        idx_b = np.where(b == 1)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return 0.0
        coords_a = np.column_stack(np.unravel_index(idx_a, (1, a.size)))
        coords_b = np.column_stack(np.unravel_index(idx_b, (1, b.size)))
        fwd = directed_hausdorff(coords_a, coords_b)[0]
        bwd = directed_hausdorff(coords_b, coords_a)[0]
        return max(fwd, bwd)

    hd95_val = hd95(pred_flat, gt_flat)

    OA = np.mean(pred_flat == gt_flat)
    return precision, recall, dice, mIoU, hd95_val, OA


def eval_net(
        net: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        pred_path: str = None,
        gt_path: str = None
):
    """
    Evaluate the model on both the fused UAV+SAT branch and the SAT-only branch.

    Returns two dicts: metrics_fusion and metrics_sat, each with keys
    ['precision','recall','dice','mIoU','hd95','OA'].
    """
    net.eval()
    n_val = len(loader)

    # Accumulators for fusion and SAT-only
    sums_f, sums_s = np.zeros(6), np.zeros(6)
    idx_f, idx_s = 0, 0

    if pred_path:
        os.makedirs(os.path.join(pred_path, "fusion"), exist_ok=True)
        os.makedirs(os.path.join(pred_path, "sat_only"), exist_ok=True)
    if gt_path:
        os.makedirs(os.path.join(gt_path, "fusion"), exist_ok=True)
        os.makedirs(os.path.join(gt_path, "sat_only"), exist_ok=True)

    with torch.no_grad(), tqdm(total=n_val, desc="Eval", leave=False) as pbar:
        for batch in loader:
            uav_img = batch["uav_image"].to(device).float()
            sat_img = batch["sat_image"].to(device).float()
            uav_mask = batch["uav_mask"].to(device).unsqueeze(1).float()
            sat_mask = batch["sat_mask"].to(device).unsqueeze(1).float()

            # --- Fusion branch ---
            out_f = net(uav_img, sat_img)
            if isinstance(out_f, tuple):
                out_f = out_f[0]
            prob_f = torch.sigmoid(out_f)
            prob_f = F.interpolate(prob_f, size=uav_mask.shape[-2:], mode='bilinear', align_corners=True)
            bam_f = (prob_f > 0.5).float()

            # compute metrics
            m_f = compute_metrics(bam_f, uav_mask)
            sums_f += np.array(m_f)

            # save masks if requested
            if pred_path:
                for img in bam_f:
                    im = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8))
                    im.save(os.path.join(pred_path, "fusion", f"{idx_f}.png"))
                    idx_f += 1
            if gt_path:
                for img in uav_mask:
                    im = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8))
                    im.save(os.path.join(gt_path, "fusion", f"{idx_f - 1}.png"))

            # --- SAT-only branch ---
            out_s = net(sat_img, sat_img)
            if isinstance(out_s, tuple):
                out_s = out_s[0]
            prob_s = torch.sigmoid(out_s)
            prob_s = F.interpolate(prob_s, size=sat_mask.shape[-2:], mode='bilinear', align_corners=True)
            bam_s = (prob_s > 0.5).float()

            m_s = compute_metrics(bam_s, sat_mask)
            sums_s += np.array(m_s)

            if pred_path:
                for img in bam_s:
                    im = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8))
                    im.save(os.path.join(pred_path, "sat_only", f"{idx_s}.png"))
                    idx_s += 1
            if gt_path:
                for img in sat_mask:
                    im = Image.fromarray((img.squeeze().cpu().numpy() * 255).astype(np.uint8))
                    im.save(os.path.join(gt_path, "sat_only", f"{idx_s - 1}.png"))

            pbar.update()

    # Compute averages
    metrics_fusion = {
        'precision': sums_f[0] / n_val,
        'recall': sums_f[1] / n_val,
        'dice': sums_f[2] / n_val,
        'mIoU': sums_f[3] / n_val,
        'hd95': sums_f[4] / n_val,
        'OA': sums_f[5] / n_val,
    }
    metrics_sat = {
        'precision': sums_s[0] / n_val,
        'recall': sums_s[1] / n_val,
        'dice': sums_s[2] / n_val,
        'mIoU': sums_s[3] / n_val,
        'hd95': sums_s[4] / n_val,
        'OA': sums_s[5] / n_val,
    }

    return metrics_fusion, metrics_sat
