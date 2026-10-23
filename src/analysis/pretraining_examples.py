"""Figure: seasonal vs long-term change in the CACo pretraining data.

Streams the beginning of the public caco10k archive (no full download), keeps the
RGB bands of the first 40 locations, and renders two locations at three dates:
two dates from the 2017-18 period and one from 2021-22. Band scaling follows
upstream seco_dataset.QUANTILES.

  python src/analysis/pretraining_examples.py --cache data/caco10k_sample
"""
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

URL = 'https://research.cs.cornell.edu/caco/data/caco/caco10k.tar.gz'
RGB = (('B4', 0, 129), ('B3', 2, 103), ('B2', 3, 88))  # band, min_q, max_q
# (location, [(date prefix, column title)...], row label)
EXAMPLES = [
    ('000238', ['20171029', '20180412', '20211018'], 'New road and housing'),
    ('001903', ['20171022', '20180709', '20211107'], 'Seasonal change only'),
]
COLUMNS = ['Period 1', 'Period 1, months later', 'Period 2, four years later']


def fetch(cache, n_locations=40):
    cache = Path(cache)
    if cache.exists() and any(cache.iterdir()):
        return
    seen = set()
    with urllib.request.urlopen(URL) as r, tarfile.open(fileobj=r, mode='r|gz') as tf:
        for m in tf:
            parts = PurePosixPath(m.name).parts
            if len(parts) >= 2:
                seen.add(parts[1])
            if len(seen) > n_locations:
                break
            if m.isfile() and len(parts) == 4 and parts[3][:-4] in {b for b, _, _ in RGB}:
                dst = cache / parts[1] / parts[2] / parts[3]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())


def rgb(folder):
    chans = [np.clip((tifffile.imread(folder / f'{b}.tif').astype(float) - lo) / (hi - lo), 0, 1)
             for b, lo, hi in RGB]
    return np.dstack(chans)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', default='data/caco10k_sample')
    args = parser.parse_args()
    fetch(args.cache)

    style.apply()
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(6.2, 4.3))
    for row, (loc, dates, label) in enumerate(EXAMPLES):
        folders = sorted((Path(args.cache) / loc).iterdir())
        for col, date in enumerate(dates):
            ax = axes[row, col]
            ax.imshow(rgb(next(f for f in folders if f.name.startswith(date))))
            ax.text(0.03, 0.04, f'{date[:4]}-{date[4:6]}-{date[6:]}', transform=ax.transAxes,
                    fontsize=7, color='white', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.45, edgecolor='none', pad=1.5))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(COLUMNS[col], fontsize=9, color=style.INK_2)
        axes[row, 0].set_ylabel(label, fontsize=9, color=style.INK)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)

    out = Path(__file__).resolve().parents[2] / 'figures'
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'pretraining_examples.{ext}')


if __name__ == '__main__':
    main()
