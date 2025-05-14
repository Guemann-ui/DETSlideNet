import os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .utils.losses import structure_loss


class TqdmLoggingHandler(logging.Handler):
    """Redirect logging through tqdm.write() so it doesn’t interfere with the progress bar."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def train_net(
        net: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int = 500,
        lr: float = 0.01,
        sat_weight: float = 0.4,
        checkpoint_dir: str = "checkpoints/",
):
    """
    Train a segmentation network with joint UAV+SAT fusion and per-epoch validation.
    sat_weight: how much the SAT-only branch contributes to the total loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # 1) Load data
            uav_img = batch["uav_image"].to(device).float()
            sat_img = batch["sat_image"].to(device).float()
            uav_mask = batch["uav_mask"].to(device).unsqueeze(1).float()
            sat_mask = batch["sat_mask"].to(device).unsqueeze(1).float()

            # 2) Forward: fusion branch
            pred_fusion = net(uav_img, sat_img)
            pred_fusion = F.interpolate(pred_fusion,
                                        size=uav_mask.shape[-2:],
                                        mode='bilinear',
                                        align_corners=True)

            # 3) Forward: SAT-only branch
            pred_sat = net(sat_img, sat_img)
            pred_sat = F.interpolate(pred_sat,
                                     size=sat_mask.shape[-2:],
                                     mode='bilinear',
                                     align_corners=True)

            # 4) Compute losses & combine
            loss_u = structure_loss(pred_fusion, uav_mask)
            loss_s = structure_loss(pred_sat, sat_mask)
            loss = (1 - sat_weight) * loss_u + sat_weight * loss_s

            # 5) Backprop & step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
            optimizer.step()

            train_loss_accum += loss.item()
            avg_loss = train_loss_accum / (batch_idx + 1)
            pbar.set_postfix(train_loss=f"{avg_loss:.4f}")

        # 6) Validation
        net.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                uav_img = batch["uav_image"].to(device).float()
                sat_img = batch["sat_image"].to(device).float()
                uav_mask = batch["uav_mask"].to(device).unsqueeze(1).float()
                sat_mask = batch["sat_mask"].to(device).unsqueeze(1).float()

                # fusion branch
                pred_fusion = net(uav_img, sat_img)
                pred_fusion = F.interpolate(pred_fusion,
                                            size=uav_mask.shape[-2:],
                                            mode='bilinear',
                                            align_corners=True)
                # SAT-only branch
                pred_sat = net(sat_img, sat_img)
                pred_sat = F.interpolate(pred_sat,
                                         size=sat_mask.shape[-2:],
                                         mode='bilinear',
                                         align_corners=True)

                loss_u = structure_loss(pred_fusion, uav_mask)
                loss_s = structure_loss(pred_sat, sat_mask)
                loss = (1 - sat_weight) * loss_u + sat_weight * loss_s

                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        logger.info(f"Epoch {epoch} ► Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")

        # 7) Checkpoint best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, "model_best.pth"))
            logger.info(f"New best model (Val Loss={best_val_loss:.4f})")

        scheduler.step()
