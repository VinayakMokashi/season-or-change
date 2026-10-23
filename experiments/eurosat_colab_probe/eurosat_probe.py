"""EuroSAT probe of the CACo authors' 100k checkpoints, as first run on Google Colab.

This protocol differs from src/probe/ in four ways: inputs are ImageNet-normalized;
the frozen backbone sits inside the trained model, so model.train() lets its
BatchNorm layers re-estimate their statistics on EuroSAT during probe training;
the batch size is 64 instead of 256; and there is one seed (42) instead of five.
Otherwise: a linear head, Adam lr 1e-3, 100 epochs, x0.1 at epochs 60 and 80.
The script prints its results; results/earlier_runs/eurosat_100k_colab_probe.json
records the printed output of our original Colab run.

  python experiments/eurosat_colab_probe/eurosat_probe.py \\
      --data_dir data/eurosat/EuroSAT_RGB --ckpt_dir checkpoints/authors_100k
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src' / 'probe'))
from datasets import EuroSAT  # noqa: E402

CKPTS = {'SeCo 100k': 'resnet18_seco_geo_100k_1000.pth', 'CACo 100k': 'resnet18_caco_geo_100k_1000.pth'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_backbone(kind, path=None):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if kind == 'imagenet' else None)
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    if kind == 'pretrain':
        backbone.load_state_dict(torch.load(path, map_location='cpu'))
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    correct = total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if optimizer:
                optimizer.zero_grad()
                criterion(logits, y).backward()
                optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt_dir', default='checkpoints/authors_100k')
    args = parser.parse_args()

    tf = T.Compose([T.Resize((64, 64)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train = DataLoader(EuroSAT(args.data_dir, 'train', tf), batch_size=64, shuffle=True, num_workers=2)
    test = DataLoader(EuroSAT(args.data_dir, 'val', tf), batch_size=64, shuffle=False, num_workers=2)

    torch.manual_seed(42)
    runs = [('Random Init', 'random', None), ('ImageNet', 'imagenet', None)] + \
           [(name, 'pretrain', Path(args.ckpt_dir) / f) for name, f in CKPTS.items()]
    for name, kind, path in runs:
        print(f'\n========== {name} ==========')
        model = nn.Sequential(load_backbone(kind, path), nn.Linear(512, 10)).to(device)
        optimizer = optim.Adam(model[-1].parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80])
        criterion = nn.CrossEntropyLoss()
        best = 0
        for epoch in range(1, 101):
            train_acc = run_epoch(model, train, criterion, optimizer)
            test_acc = run_epoch(model, test, criterion)
            scheduler.step()
            best = max(best, test_acc)
            if epoch % 10 == 0:
                print(f'  Epoch {epoch:3d}/100 | Train Acc: {train_acc:.2f}% | Val Acc: {test_acc:.2f}%')
        print(f'  => Best Val Acc: {best:.2f}%')


if __name__ == '__main__':
    main()
