"""Summarise the pretraining logs and plot per-method loss curves.

Reads results/pretraining/<method>_metrics.csv (written by src/pretrain/train.py)
and writes results/pretraining/summary.json and figures/pretraining_loss.{pdf,png}.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / 'results' / 'pretraining'
METHODS = {'seco': 'SeCo', 'teco': 'TeCo', 'caco': 'CACo', 'moco': 'MoCo'}

# Dataset items per epoch for 2,000 locations x 10 dates (third_party/caco_upstream
# datasets): SeCo one item per location, TeCo/CACo one per (location, period),
# MoCo one per image. Batches of 256 with drop_last.
ITEMS_PER_EPOCH = {'seco': 2000, 'teco': 4000, 'caco': 4000, 'moco': 20000}
BATCH_SIZE = 256
LR_DROPS = (250, 350)


def load(method):
    df = pd.read_csv(LOGS / f'{method}_metrics.csv')
    # CACo was resumed from its epoch-299 checkpoint; keep the resumed rows.
    curve = df.drop_duplicates('epoch', keep='last').sort_values('epoch')
    return df, curve


def main():
    style.apply()
    summary = {}
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
    for ax, (key, name) in zip(axes, METHODS.items()):
        raw, curve = load(key)
        steps_per_epoch = ITEMS_PER_EPOCH[key] // BATCH_SIZE
        epochs = int(curve['epoch'].max()) + 1
        summary[name] = {
            'epochs': epochs,
            'optimizer_steps': steps_per_epoch * epochs,
            'gpu_hours': round(raw['epoch_time_s'].sum() / 3600, 2),
            'loss_first_epoch': float(curve['loss'].iloc[0]),
            'loss_last_10_epochs': round(float(curve['loss'].tail(10).mean()), 3),
        }
        for x in LR_DROPS:
            ax.axvline(x, color=style.AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
        ax.plot(curve['epoch'], curve['loss'], color=style.COLORS[name], linewidth=1.6)
        ax.set_title(name, loc='left', fontsize=10, color=style.INK, fontweight='semibold')
        ax.text(0.98, 0.95, f'{summary[name]["optimizer_steps"]:,} steps\n'
                            f'{summary[name]["gpu_hours"]:.1f} GPU-h',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color=style.INK_2,
                bbox=dict(facecolor=style.SURFACE, edgecolor='none', pad=1.5))
        ax.set_xlabel('epoch')
        ax.set_xlim(0, 400)
    axes[0].set_ylabel('contrastive loss')
    fig.tight_layout(w_pad=1.5)

    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'pretraining_loss.{ext}')
    (LOGS / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
