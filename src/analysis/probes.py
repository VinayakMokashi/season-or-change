"""Tables, paired comparisons, hypothesis verdicts and figures for the probe sweep.

Reads results/probes/<dataset>/<backbone>.json and predictions/<backbone>.npz
(written by src/probe/run_probes.py) and writes
  results/probes/summary.md        accuracy tables and paired comparisons
  results/probes/comparisons.json  every paired difference with its 95% CI
  results/probes/hypotheses.json   verdicts for docs/HYPOTHESES.md
  figures/probe_accuracy.{pdf,png}, figures/component_steps.{pdf,png}
"""
import json
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / 'results' / 'probes'
DATASETS = {'eurosat': 'EuroSAT', 'ucmerced': 'UC Merced'}
N_BOOT = 10_000

# (backbone, method, group) in display order
ROWS = [
    ('random', 'Random', 'Baselines'),
    ('imagenet', 'ImageNet', 'Baselines'),
    ('seco-10k', 'SeCo', 'Our pretraining (2k locations, 400 ep)'),
    ('teco-10k', 'TeCo', 'Our pretraining (2k locations, 400 ep)'),
    ('caco-10k', 'CACo', 'Our pretraining (2k locations, 400 ep)'),
    ('moco-100k', 'MoCo', "Authors' checkpoints (20k locations)"),
    ('seco-100k', 'SeCo', "Authors' checkpoints (20k locations)"),
    ('teco-100k', 'TeCo', "Authors' checkpoints (20k locations)"),
    ('caco-100k', 'CACo', "Authors' checkpoints (20k locations)"),
]
PAIRS = [
    ('teco-10k', 'seco-10k'), ('caco-10k', 'teco-10k'), ('caco-10k', 'seco-10k'),
    ('teco-100k', 'seco-100k'), ('caco-100k', 'teco-100k'), ('caco-100k', 'seco-100k'),
    ('seco-100k', 'moco-100k'), ('caco-100k', 'moco-100k'),
    ('imagenet', 'caco-100k'), ('imagenet', 'moco-100k'), ('imagenet', 'teco-10k'),
]


def load(dataset):
    out = {}
    for name, _, _ in ROWS:
        f = PROBES / dataset / f'{name}.json'
        if f.exists():
            s = json.loads(f.read_text(encoding='utf-8'))
            s['correct'] = np.load(PROBES / dataset / 'predictions' / f'{name}.npz')['seed_avg_correct']
            out[name] = s
    return out


def paired(a, b, key):
    """Mean difference (points) and 95% paired-bootstrap interval, seeded per comparison."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    diff = a - b
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = diff[idx].mean(1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean() * 100), float(lo), float(hi)


def acc(s):
    return s['linear_probe']['final_acc_mean']


def verdict_chain(steps):
    """Supported if every step is resolved and positive; refuted if any is resolved negative."""
    if any(hi < 0 for _, lo, hi in steps):
        return 'refuted'
    if all(lo > 0 for _, lo, hi in steps):
        return 'supported'
    return 'unresolved'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    results = {ds: load(ds) for ds in DATASETS}
    comparisons = {}
    for ds, res in results.items():
        comparisons[ds] = {}
        for a, b in PAIRS:
            if a in res and b in res:
                d, lo, hi = paired(res[a]['correct'], res[b]['correct'], f'{ds}:{a}-{b}')
                comparisons[ds][f'{a} - {b}'] = {'diff': d, 'ci95': [lo, hi]}
    (PROBES / 'comparisons.json').write_text(json.dumps(comparisons, indent=1), encoding='utf-8')

    def step(ds, a, b):
        c = comparisons[ds].get(f'{a} - {b}')
        return (c['diff'], *c['ci95']) if c else None

    hyp = {}
    s1 = [step('eurosat', 'teco-100k', 'seco-100k'), step('eurosat', 'caco-100k', 'teco-100k')]
    hyp['H1'] = {'steps': s1, 'verdict': verdict_chain(s1) if all(s1) else 'missing'}
    s2 = [step('eurosat', 'seco-100k', 'moco-100k'), step('eurosat', 'caco-100k', 'moco-100k')]
    hyp['H2'] = {'steps': s2, 'verdict': verdict_chain(s2) if all(s2) else 'missing'}
    s3 = step('eurosat', 'caco-10k', 'teco-10k')
    if s3:
        d, lo, hi = s3
        v = 'supported' if (d < 1.0 and lo <= 0 <= hi) else ('refuted' if lo > 0 and d >= 1.0 else 'unresolved')
        hyp['H3'] = {'step': s3, 'verdict': v}
    gaps = {}
    for ds, res in results.items():
        # Every satellite-pretrained encoder probed at each scale. Our MoCo v2 encoder
        # is only in results/original_eval (single seed, other UC Merced split), so the
        # 10k comparison covers SeCo/TeCo/CACo; the report discusses MoCo separately.
        for scale, names in (('10k', ['seco-10k', 'teco-10k', 'caco-10k']),
                             ('100k', ['moco-100k', 'seco-100k', 'teco-100k', 'caco-100k'])):
            best = max((n for n in names if n in res), key=lambda n: acc(res[n]), default=None)
            if best and 'imagenet' in res:
                gaps[f'{ds}/{scale}'] = {'best_satellite': best, 'imagenet_minus_best': acc(res['imagenet']) - acc(res[best])}
    h4 = all(gaps.get(f'ucmerced/{s}', {}).get('imagenet_minus_best', -1e9) >
             gaps.get(f'eurosat/{s}', {}).get('imagenet_minus_best', 1e9) for s in ('10k', '100k'))
    hyp['H4'] = {'gaps': gaps, 'verdict': 'supported' if h4 else 'not supported'}
    (PROBES / 'hypotheses.json').write_text(json.dumps(hyp, indent=1), encoding='utf-8')

    lines = ['# Linear-probe sweep', '',
             'Final-epoch top-1 accuracy (%), mean ± s.d. over 5 probe seeds; frozen ResNet-18 features.', '',
             '| Backbone | ' + ' | '.join(f'{v} linear | {v} kNN-5 | {v} P@20' for v in DATASETS.values()) + ' |',
             '|---|' + '---:|' * 3 * len(DATASETS)]
    for name, method, group in ROWS:
        cells = []
        for ds in DATASETS:
            s = results[ds].get(name)
            cells += ([f'{acc(s):.2f} ± {s["linear_probe"]["final_acc_std"]:.2f}', f'{s["knn"]["acc"]:.2f}',
                       f'{s["retrieval"]["P@20"]:.2f}'] if s else ['–'] * 3)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired differences (points, 95% paired-bootstrap interval)', '',
              '| Comparison | ' + ' | '.join(DATASETS.values()) + ' |', '|---|' + '---:|' * len(DATASETS)]
    for a, b in PAIRS:
        cells = []
        for ds in DATASETS:
            c = comparisons[ds].get(f'{a} - {b}')
            cells.append(f'{c["diff"]:+.2f} [{c["ci95"][0]:+.2f}, {c["ci95"][1]:+.2f}]' if c else '–')
        lines.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Hypotheses (docs/HYPOTHESES.md)', '']
    lines += [f'- **{k}**: {v["verdict"]}' for k, v in hyp.items()]
    (PROBES / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

    plot(results)
    plot_steps(comparisons)
    # Smaller canvases (relatively larger text) for the poster.
    plot(results, figsize=(5.6, 3.3), out_name='probe_accuracy_poster')
    plot_steps(comparisons, figsize=(5.6, 2.4), out_name='component_steps_poster')


def plot(results, figsize=(7.2, 3.6), out_name='probe_accuracy'):
    style.apply()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=figsize, sharey=True)
    y, ticks, labels, group_pos = 0, [], [], []
    positions = {}
    last_group = None
    for name, method, group in ROWS:
        if group != last_group:
            if last_group is not None:
                y += 0.6
            group_pos.append((y, group))
            last_group = group
        positions[name] = y
        ticks.append(y)
        labels.append(method)
        y += 1
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        for name, method, _ in ROWS:
            s = results[ds].get(name)
            if not s:
                continue
            a = acc(s)
            ax.barh(positions[name], a, height=0.72, color=style.COLORS[method], edgecolor=style.SURFACE,
                    linewidth=1)
            ax.text(a + 0.8, positions[name], f'{a:.1f}', va='center', fontsize=7.5, color=style.INK_2)
        ax.set_title(title, loc='left', fontsize=10, fontweight='semibold')
        ax.set_xlim(0, 100)
        ax.set_xlabel('linear-probe accuracy (%)')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=style.GRID, linewidth=0.6)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for pos, group in group_pos:
        axes[0].annotate(group, xy=(0, pos - 0.62), xycoords=('axes fraction', 'data'),
                         fontsize=7.5, color=style.MUTED, ha='left', va='bottom')
    fig.tight_layout(w_pad=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(ROOT / 'figures' / f'{out_name}.{ext}')


# Paper Table 6 (ResNet-18, 100k, EuroSAT linear probe): +Improved sampling 91.42,
# +Long-term temporal contrast 92.56, +Change-awareness 93.08.
PAPER_STEPS = {'long': 92.56 - 91.42, 'change': 93.08 - 92.56}
STEPS = [('long', 'Long-term contrast\n(TeCo − SeCo)', 'teco', 'seco'),
         ('change', 'Change-awareness\n(CACo − TeCo)', 'caco', 'teco')]
SCALES = [('10k', 'Our pretraining (2k locations)', 'o', style.MUTED),
          ('100k', "Authors' checkpoints (20k locations)", 's', style.INK)]


def plot_steps(comparisons, figsize=(7.2, 2.5), out_name='component_steps'):
    """Each component's contribution with its 95% interval, at both scales."""
    style.apply()
    fig, axes = plt.subplots(1, len(DATASETS), figsize=figsize, sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS.items()):
        ax.axvline(0, color=style.AXIS, linewidth=0.8, zorder=0)
        for i, (key, _, a, b) in enumerate(STEPS):
            for j, (scale, label, marker, color) in enumerate(SCALES):
                c = comparisons[ds].get(f'{a}-{scale} - {b}-{scale}')
                if not c:
                    continue
                y = i + (j - 0.5) * 0.32
                lo, hi = c['ci95']
                ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle='round')
                ax.plot(c['diff'], y, marker=marker, color=color, markersize=7, linestyle='none',
                        markeredgecolor=style.SURFACE, markeredgewidth=1.5,
                        label=label if (i == 0 and ds == 'eurosat') else None)
            if ds == 'eurosat':
                ax.plot(PAPER_STEPS[key], i + 0.16, marker='|', color=style.PAPER, markersize=16,
                        markeredgewidth=2, linestyle='none',
                        label='Paper, 100k (Table 6)' if i == 0 else None)
        ax.set_title(title, loc='left', fontsize=10, fontweight='semibold')
        ax.set_xlabel('accuracy difference (points)')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=style.GRID, linewidth=0.6)
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(STEPS)))
    axes[0].set_yticklabels([s[1] for s in STEPS])
    axes[0].set_ylim(len(STEPS) - 0.4, -0.6)
    fig.legend(loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.55, -0.08), handletextpad=0.3)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(ROOT / 'figures' / f'{out_name}.{ext}')


if __name__ == '__main__':
    main()
