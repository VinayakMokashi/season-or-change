"""ResNet-18 feature extractors: random init, ImageNet, or a pretrained encoder file."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

# name -> encoder file, relative to --ckpt_dir. `random` and `imagenet` need no file.
BACKBONES = {
    'random': None,
    'imagenet': None,
    'seco-10k': 'ours_10k/resnet18_seco_geo_10k_400.pth',
    'teco-10k': 'ours_10k/resnet18_teco_geo_10k_400.pth',
    'caco-10k': 'ours_10k/resnet18_caco_geo_10k_400.pth',
    'moco-10k': 'ours_10k/resnet18_moco_geo_10k_400.pth',  # not distributed; probed if present
    'moco-100k': 'authors_100k/resnet18_moco_geo_100k_200.pth',
    'seco-100k': 'authors_100k/resnet18_seco_geo_100k_1000.pth',
    'teco-100k': 'authors_100k/resnet18_teco_geo_100k_1000.pth',
    'caco-100k': 'authors_100k/resnet18_caco_geo_100k_1000.pth',
}

RANDOM_INIT_SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_backbone(name, ckpt_dir):
    """Return (frozen eval-mode backbone, provenance dict)."""
    if name == 'random':
        torch.manual_seed(RANDOM_INIT_SEED)
        model = resnet18(weights=None)
    elif name == 'imagenet':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    info = {'name': name, 'file': None, 'sha256': None, 'bn_steps': None}
    rel = BACKBONES[name]
    if rel is not None:
        path = Path(ckpt_dir) / rel
        state = torch.load(path, map_location='cpu')
        backbone.load_state_dict(state, strict=True)
        steps = {int(v) for k, v in state.items() if k.endswith('num_batches_tracked')}
        info.update(file=Path(rel).name, sha256=sha256(path), bn_steps=sorted(steps))

    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.eval(), info
