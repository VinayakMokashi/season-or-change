"""BigEarthNet frozen-backbone probe of the CACo authors' released 100k checkpoints.

Trains an MLP head (512 -> 1024 -> 19, dropout 0.2) on frozen ResNet-18 features
with 10% of the training labels, and reports micro-averaged average precision (mAP) on
the validation split every 10 epochs. This is a cleaned version of the script
that wrote results/earlier_runs/bigearthnet_100k_probe.log: same settings, with
paths as arguments and the final comparison table and plot removed. The
validation loader drops its last partial batch (123,648 of 123,723 patches are
scored). Pass --download to fetch BigEarthNet-S2 v1.0 (about 70 GB) on first use.

  python experiments/bigearthnet/train_bigearthnet.py --data_dir data/bigearthnet \\
      --ckpt_dir checkpoints/authors_100k
"""
import argparse
import os
import sys
from itertools import chain
from pathlib import Path
from urllib import request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'caco_upstream' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.pretrained_checkpoints import saved_torch_ckpts  # noqa: E402
from bigearthnet_dataset import Bigearthnet  # noqa: E402

CKPT_URL = 'https://research.cs.cornell.edu/caco/checkpoints/'
TRAIN_FRAC = 0.1
NUM_EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-5
MILESTONES = [60, 80]
FREEZE_BACKBONE = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigEarthNetClassifier(nn.Module):
    def __init__(self, backbone, in_features=512, num_classes=19,
                 hidden_dim=1024, dropout=0.2, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_backbone):
            feats = self.backbone(x)
        return self.head(feats)


def download_checkpoint(ckpt_dir, ckpt_name):
    dest = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.isfile(dest):
        print(f'Downloading {ckpt_name} ...')
        request.urlretrieve(CKPT_URL + ckpt_name, dest)


def load_backbone(backbone_type, ckpt_dir, ckpt_key=None):
    model = resnet18(pretrained=False)
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    if backbone_type == 'imagenet':
        model = resnet18(pretrained=True)
        backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif backbone_type == 'pretrain':
        state_dict = torch.load(os.path.join(ckpt_dir, saved_torch_ckpts[ckpt_key]), map_location='cpu')
        result = backbone.load_state_dict(state_dict, strict=False)
        print(f'Loaded: {saved_torch_ckpts[ckpt_key]}')
        print(f'Missing: {result.missing_keys} | Unexpected: {result.unexpected_keys}')
    return backbone


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    model.backbone.eval()  # frozen backbone keeps its BatchNorm statistics
    total_loss, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        all_preds.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        all_targets.append(y.numpy())
    return average_precision_score(np.concatenate(all_targets), np.concatenate(all_preds),
                                   average='micro') * 100.0


def run_experiment(name, backbone_type, ckpt_dir, loaders, ckpt_key=None):
    print(f'\n========== {name} ==========')
    train_loader, val_loader = loaders
    backbone = load_backbone(backbone_type, ckpt_dir, ckpt_key).to(device)
    model = BigEarthNetClassifier(backbone, freeze_backbone=FREEZE_BACKBONE).to(device)
    params = chain(model.backbone.parameters(), model.head.parameters()) if not FREEZE_BACKBONE \
        else model.head.parameters()
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.MultiLabelSoftMarginLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    best_map = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        if epoch % 10 == 0:
            val_map = evaluate(model, val_loader)
            best_map = max(best_map, val_map)
            print(f'  Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Val mAP: {val_map:.2f}')
    print(f'  => Best Val mAP: {best_map:.2f}')
    return best_map


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt_dir', default='checkpoints/authors_100k')
    parser.add_argument('--download', action='store_true', help='download BigEarthNet into --data_dir')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for key in ['r18_100k_caco', 'r18_100k_seco']:
        download_checkpoint(args.ckpt_dir, saved_torch_ckpts[key])

    transform = T.Compose([T.Resize((128, 128), interpolation=Image.BICUBIC), T.ToTensor()])
    full_train = Bigearthnet(root=args.data_dir, split='train', transform=transform, download=args.download)
    val_dataset = Bigearthnet(root=args.data_dir, split='val', transform=transform, download=args.download)
    np.random.seed(42)
    train_indices = np.random.choice(len(full_train), int(len(full_train) * TRAIN_FRAC), replace=False)
    train_dataset = Subset(full_train, train_indices)
    loaders = (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True),
    )
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    torch.manual_seed(42)
    for name, btype, key in [('Random Init', 'random', None), ('ImageNet', 'imagenet', None),
                             ('SeCo 100k', 'pretrain', 'r18_100k_seco'),
                             ('CACo 100k', 'pretrain', 'r18_100k_caco')]:
        run_experiment(name, btype, args.ckpt_dir, loaders, key)


if __name__ == '__main__':
    main()
