"""Figure: UC Merced test images where ImageNet features are right and CACo is wrong.

Uses the per-image predictions saved by src/probe/run_probes.py. An image is shown
if the linear probe on ImageNet features classifies it correctly for every seed
and the probe on the authors' CACo checkpoint misclassifies it for every seed;
the caption gives CACo's prediction from the first seed.

  python src/analysis/failure_examples.py --ucmerced data/ucmerced/UCMerced_LandUse/Images
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / 'results' / 'probes' / 'ucmerced'
N_SHOW = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ucmerced', default='data/ucmerced/UCMerced_LandUse/Images')
    args = parser.parse_args()

    files = (ROOT / 'splits' / 'ucmerced_test.txt').read_text(encoding='utf-8').split()
    classes = json.loads((PRED / 'caco-100k.json').read_text(encoding='utf-8'))['classes']
    caco = np.load(PRED / 'predictions' / 'caco-100k.npz')
    imnet = np.load(PRED / 'predictions' / 'imagenet.npz')
    labels = caco['labels']
    picks = np.where((imnet['seed_avg_correct'] == 1) & (caco['seed_avg_correct'] == 0))[0]

    # One image per true class, in test-set order, for variety.
    chosen, seen = [], set()
    for i in picks:
        if labels[i] not in seen:
            chosen.append(i)
            seen.add(labels[i])
        if len(chosen) == N_SHOW:
            break

    # Report version, and a smaller canvas (relatively larger text) for the poster.
    for width, out_name in ((1.45, 'failure_examples'), (1.15, 'failure_examples_poster')):
        style.apply()
        fig, axes = plt.subplots(1, len(chosen), figsize=(width * len(chosen), width * 1.35))
        for ax, i in zip(np.atleast_1d(axes), chosen):
            ax.imshow(Image.open(Path(args.ucmerced) / files[i]).convert('RGB'))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(classes[labels[i]], fontsize=8, color=style.INK)
            ax.set_xlabel(f'CACo: {classes[caco["linear_pred"][0][i]]}', fontsize=7.5, color=style.INK_2)
        fig.tight_layout(w_pad=0.4)
        for ext in ('pdf', 'png'):
            fig.savefig(ROOT / 'figures' / f'{out_name}.{ext}')
        plt.close(fig)
    print(f'{len(picks)} images: ImageNet right in all seeds, CACo wrong in all seeds')


if __name__ == '__main__':
    main()
