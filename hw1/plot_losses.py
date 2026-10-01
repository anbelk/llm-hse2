"""Build training-loss plots: python hw1/plot_losses.py (requires matplotlib)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'training'
PRIMARY = ['baseline', 'batch8', 'accumulation2', 'lr1e-4', 'cosine',
           'optimizer_unfused', 'compile_off', 'tf32_off']
PRECISION = ['bf16_amp_sdpa', 'fp32_sdpa']


def load(name):
    with (ROOT / 'losses' / f'{name}.csv').open() as source:
        return [(float(row['elapsed_seconds']), float(row['loss']))
                for row in csv.DictReader(source)]


def smooth(rows):
    """Mean loss and time per 20-second bin; include final-step overshoot."""
    bins = {}
    for seconds, loss in rows:
        bins.setdefault(min(44, int(seconds // 20)), []).append((seconds, loss))
    return [(sum(t for t, _ in values) / len(values),
             sum(loss for _, loss in values) / len(values))
            for _, values in sorted(bins.items())]


def style(ax, title):
    ax.set(title=title, xlabel='Elapsed training time (s)',
           ylabel='Training cross-entropy loss', xlim=(0, 900))
    ax.grid(alpha=0.25)


def comparison(names, filename, title):
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for name in names:
        points = smooth(load(name))
        ax.plot(*zip(*points), label=name, linewidth=1.8)
    style(ax, title)
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(ROOT / filename, dpi=180)
    plt.close(fig)


def main():
    comparison(PRIMARY, 'comparison_plots.png',
               'Primary experiments: training loss (20-second means)')
    comparison(PRECISION, 'precision_comparison.png',
               'BF16 autocast vs FP32: training loss (20-second means)')
    names = PRIMARY + PRECISION
    fig, axes = plt.subplots(5, 2, figsize=(14, 16), layout='constrained')
    for ax, name in zip(axes.flat, names):
        rows = load(name)
        ax.plot(*zip(*rows), alpha=0.2, linewidth=0.5, color='#4775b3')
        ax.plot(*zip(*smooth(rows)), linewidth=1.6, color='#152d54')
        style(ax, name)
    fig.savefig(ROOT / 'individual_loss_curves.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
