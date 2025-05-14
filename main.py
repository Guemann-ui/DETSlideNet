import warnings

warnings.filterwarnings('ignore')

import argparse
import logging
import sys
import os

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataloader import CASLDataset
from src.nets import DETSLideNet
from src.train import train_net
from src.test import eval_net


def get_args():
    parser = argparse.ArgumentParser(
        description='Train or test the DETSLideNet model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-m', '--mode', type=str, default='test',
                        help='Mode: "train" or "test"')
    parser.add_argument('-r', '--region', type=str,
                        default='Moxitaidi,TiburonPeninsula,Jiuzhaivalley,Moxitown,LongxiRiver',
                        help='Comma-separated list of regions')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('-w', '--sat-weight', type=float, default=0.4,
                        help='Weight for SAT-only loss in [0,1]')
    parser.add_argument('-f', '--load', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('-s', '--img_size', type=int, default=128,
                        help='Input image size (H and W)')
    return parser.parse_args()


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger()

    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    base_root = './data/data/'
    data_path = os.path.join(base_root, args.mode)
    regions = args.region.split(',')

    if args.mode == 'train':
        # -- Prepare datasets --
        dataset = CASLDataset(base_root=data_path, regions=regions, input_size=args.img_size)
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                batch_size=1, shuffle=False)

        logger.info(f'''
        Training mode
          Regions:         {regions}
          Epochs:          {args.epochs}
          Batch size:      {args.batch_size}
          LR:              {args.learning_rate}
          SAT weight:      {args.sat_weight}
          Train size:      {len(train_idx)}
          Val size:        {len(val_idx)}
          Image size:      {args.img_size}
        ''')

        net = DETSLideNet(dim=args.img_size, n_class=1).to(device)
        if args.load:
            net.load_state_dict(torch.load(args.load, map_location=device))
            logger.info(f'Loaded pretrained model from {args.load}')

        try:
            train_net(
                net=net,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.learning_rate,
                sat_weight=args.sat_weight,
                checkpoint_dir="checkpoints/"
            )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), "checkpoints/INTERRUPTED.pth")
            logger.info('Training interrupted; model saved to INTERRUPTED.pth')
            sys.exit()

    elif args.mode == 'test':
        # -- Prepare test set --
        testset = CASLDataset(base_root=data_path, regions=regions, input_size=args.img_size)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        logger.info(f'''
        Test mode
          Regions:      {regions}
          Test size:    {len(testset)}
          Image size:   {args.img_size}
        ''')
        if not args.load:
            raise ValueError("In 'test' mode you must provide --load <checkpoint>")

        net = DETSLideNet(dim=args.img_size, n_class=1).to(device)
        net.load_state_dict(torch.load(args.load, map_location=device))
        logger.info(f'Loaded model from {args.load}')

        pred_dir = os.path.join(data_path, f'pred_{args.region}')
        gt_dir = os.path.join(data_path, f'label_{args.region}')
        metrics_fusion, metrics_sat = eval_net(net, test_loader, device,
                                               pred_path=pred_dir,
                                               gt_path=gt_dir)

        # -- Report results --
        logger.info("=== Fusion branch metrics ===")
        for k, v in metrics_fusion.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=== SAT-only branch metrics ===")
        for k, v in metrics_sat.items():
            logger.info(f"  {k}: {v:.4f}")

    else:
        raise ValueError("Mode must be 'train' or 'test'")
