"""Frozen-feature evaluation of ResNet-18 encoders on EuroSAT and UC Merced.

For every backbone this script:
  1. extracts 512-d features for the train and test splits (cached to --cache_dir);
  2. trains a linear probe once per seed (Adam lr 1e-3, 100 epochs,
     x0.1 at epochs 60 and 80, batch 256) and records final-epoch accuracy;
  3. computes k-NN (k=5, cosine) accuracy and retrieval precision on the test split;
  4. writes a JSON summary and per-image predictions to --out_dir/<dataset>/.

Example:
  python src/probe/run_probes.py --dataset eurosat --data_dir data/eurosat/EuroSAT_RGB \\
      --ckpt_dir checkpoints --out_dir results/probes
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbones import BACKBONES, RANDOM_INIT_SEED, build_backbone  # noqa: E402
from datasets import EuroSAT, UCMerced  # noqa: E402

EPOCHS = 100
MILESTONES = (60, 80)
BATCH_SIZE = 256
KNN_K = 5
RETRIEVAL_KS = (1, 5, 10, 20, 100)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_splits(args):
    if args.dataset == 'eurosat':
        tf = T.ToTensor()
        train = EuroSAT(args.data_dir, 'train', tf)
        test = EuroSAT(args.data_dir, 'val', tf)
    else:
        # A few UC Merced images are not exactly 256x256; resize so batches collate.
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        train = UCMerced(args.data_dir, Path(args.split_dir) / 'ucmerced_train.txt', tf)
        test = UCMerced(args.data_dir, Path(args.split_dir) / 'ucmerced_test.txt', tf)
    return train, test


@torch.no_grad()
def extract(backbone, dataset, workers):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers)
    feats, labels = [], []
    for x, y in loader:
        feats.append(backbone(x).numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def cache_key(info, train, test):
    """Hash of the weights (or backbone name), random-init seed and exact image lists."""
    h = hashlib.sha256((info['sha256'] or f'{info["name"]}-{RANDOM_INIT_SEED}').encode())
    for ds in (train, test):
        h.update('\n'.join(f'{p}:{t}' for p, t in ds.samples).encode())
    return h.hexdigest()[:16]


def features(name, info, dataset_name, train, test, args):
    cache = Path(args.cache_dir) / dataset_name / f'{name}_{cache_key(info, train, test)}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['xtr'], z['ytr'], z['xte'], z['yte']
    backbone, _ = build_backbone(name, args.ckpt_dir)
    t0 = time.time()
    xtr, ytr = extract(backbone, train, args.workers)
    xte, yte = extract(backbone, test, args.workers)
    log(f'  features {xtr.shape} / {xte.shape} in {time.time() - t0:.0f}s')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


def linear_probe(xtr, ytr, xte, yte, num_classes, seed):
    torch.manual_seed(seed)
    clf = nn.Linear(xtr.shape[1], num_classes)
    opt = optim.Adam(clf.parameters())
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=list(MILESTONES))
    loss_fn = nn.CrossEntropyLoss()
    Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr).long()
    Xte, Yte = torch.from_numpy(xte), torch.from_numpy(yte).long()

    history = []
    for _ in range(EPOCHS):
        clf.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            loss = loss_fn(clf(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(Xte).argmax(1)
        history.append(float((pred == Yte).float().mean()))
    return pred.numpy(), history


def knn(xtr, ytr, xte, k=KNN_K):
    tr = F.normalize(torch.from_numpy(xtr), dim=1)
    te = F.normalize(torch.from_numpy(xte), dim=1)
    nn_idx = (te @ tr.T).topk(k, dim=1).indices
    return torch.mode(torch.from_numpy(ytr)[nn_idx], dim=1).values.numpy()


def retrieval(xte, yte):
    f = F.normalize(torch.from_numpy(xte), dim=1)
    sim = f @ f.T
    sim.fill_diagonal_(-1e9)  # a query never retrieves itself
    order = sim.argsort(dim=1, descending=True)[:, :max(RETRIEVAL_KS)].numpy()
    rel = (yte[order] == yte[:, None]).astype(np.float64)
    n_rel = np.bincount(yte)[yte] - 1
    out = {}
    for k in RETRIEVAL_KS:
        out[f'P@{k}'] = 100 * rel[:, :k].mean()
        out[f'R@{k}'] = 100 * (rel[:, :k].sum(1) / n_rel).mean()
    return out


def per_class_recall(pred, y, num_classes):
    return [float((pred[y == c] == c).mean()) for c in range(num_classes)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['eurosat', 'ucmerced'], required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split_dir', default='splits', help='UC Merced split files')
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--out_dir', default='results/probes')
    parser.add_argument('--cache_dir', default='cache/features')
    parser.add_argument('--backbones', nargs='*', default=list(BACKBONES))
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    train, test = load_splits(args)
    num_classes = len(train.classes)
    out_dir = Path(args.out_dir) / args.dataset
    (out_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    log(f'{args.dataset}: train={len(train)} test={len(test)} classes={num_classes} '
        f'threads={torch.get_num_threads()}')

    for name in args.backbones:
        rel = BACKBONES[name]
        if rel is not None and not (Path(args.ckpt_dir) / rel).exists():
            log(f'== {name}: skipped, {Path(args.ckpt_dir) / rel} not found')
            continue
        log(f'== {name}')
        _, info = build_backbone(name, args.ckpt_dir)
        xtr, ytr, xte, yte = features(name, info, args.dataset, train, test, args)

        preds, runs, history0 = [], [], None
        for seed in args.seeds:
            pred, history = linear_probe(xtr, ytr, xte, yte, num_classes, seed)
            preds.append(pred)
            runs.append({'seed': seed, 'final_acc': 100 * history[-1],
                         'best_acc': 100 * max(history), 'best_epoch': int(np.argmax(history))})
            if history0 is None:
                history0 = [100 * h for h in history]
        finals = np.array([r['final_acc'] for r in runs])
        knn_pred = knn(xtr, ytr, xte)
        seed_avg_correct = (np.stack(preds) == yte[None]).mean(0)

        summary = {
            'dataset': args.dataset,
            'backbone': info,
            'n_train': int(len(ytr)), 'n_test': int(len(yte)),
            'classes': train.classes,
            'linear_probe': {
                'final_acc_mean': float(finals.mean()),
                'final_acc_std': float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
                'runs': runs,
                'history_first_seed': history0,
                'per_class_recall': [float(np.mean([r[c] for r in
                                     (per_class_recall(p, yte, num_classes) for p in preds)]))
                                     for c in range(num_classes)],
            },
            'knn': {'k': KNN_K, 'acc': 100 * float((knn_pred == yte).mean()),
                    'correct': int((knn_pred == yte).sum())},
            'retrieval': retrieval(xte, yte),
            'protocol': {'input': 'ToTensor' + (' after Resize(256)' if args.dataset == 'ucmerced' else ''),
                         'backbone_mode': 'eval (frozen)', 'optimizer': 'Adam lr=1e-3',
                         'epochs': EPOCHS, 'milestones': list(MILESTONES), 'batch_size': BATCH_SIZE},
            'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                            'device': 'cpu'},
        }
        lp = summary['linear_probe']
        log(f'  linear {lp["final_acc_mean"]:.2f} ± {lp["final_acc_std"]:.2f} | '
            f'kNN {summary["knn"]["acc"]:.2f} | P@20 {summary["retrieval"]["P@20"]:.2f}')

        (out_dir / f'{name}.json').write_text(json.dumps(summary, indent=1), encoding='utf-8', newline='\n')
        np.savez_compressed(out_dir / 'predictions' / f'{name}.npz', labels=yte,
                            linear_pred=np.stack(preds), seeds=np.array(args.seeds),
                            knn_pred=knn_pred, seed_avg_correct=seed_avg_correct)


if __name__ == '__main__':
    main()
