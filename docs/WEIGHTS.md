# Encoder weights

`scripts/download.py` fetches every file below into `checkpoints/`.

## Our encoders

ResNet-18 query encoders pretrained for 400 epochs on the 2,000-location CACo
archive (`clean_10k_geography`). Download them from the
[v1.0 GitHub release](https://github.com/VinayakMokashi/season-or-change/releases/tag/v1.0),
or from the [Google Drive mirror](https://drive.google.com/drive/folders/1--bZxzBoeUYKOH7TfrxEeI06rUeX1Qj5).

| File | Method | Optimizer steps | SHA-256 (first 16) |
|---|---|---|---|
| `resnet18_seco_geo_10k_400.pth` | SeCo | 2,800 | `e19e1d4aa89f0407` |
| `resnet18_teco_geo_10k_400.pth` | TeCo | 6,000 | `9414c6fb35ac97ac` |
| `resnet18_caco_geo_10k_400.pth` | CACo | 6,000 | `c99978cb1359d1fc` |

Our MoCo v2 encoder is not distributed; its training log and original
evaluation are in `results/`.

## The CACo authors' encoders

These are released by Mall et al. at <https://research.cs.cornell.edu/caco/checkpoints/>:

- `resnet18_moco_geo_100k_200.pth`
- `resnet18_seco_geo_100k_1000.pth`
- `resnet18_teco_geo_100k_1000.pth`
- `resnet18_caco_geo_100k_1000.pth`

## Loading

Each file is a plain `state_dict` for a ResNet-18 without its classification
layer:

```python
import torch
from torch import nn
from torchvision.models import resnet18

backbone = nn.Sequential(*list(resnet18().children())[:-1], nn.Flatten())
backbone.load_state_dict(torch.load('checkpoints/ours_10k/resnet18_caco_geo_10k_400.pth', map_location='cpu'))
features = backbone.eval()(torch.rand(8, 3, 64, 64))   # (8, 512)
```

The pretraining inputs were ImageNet-normalized RGB composites (bands B4, B3, B2).
The main evaluation (`src/probe/`), like the authors' EuroSAT input pipeline,
feeds `ToTensor()` images in [0, 1]; the Colab probe in `experiments/`
normalizes them. `scripts/download.py` checks every file against its full
SHA-256 hash.
