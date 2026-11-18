# Held-out eval for the synthetic silent-command decoder.
# Run from the repo root: python3 sim/decode_eval.py
# Re-run and replace demos/confusion_matrix.png if the generator or features change.
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from decode import SYNTHETIC_NOTE, evaluate, format_report


def save_confusion_png(result, path):
    labels = result['labels']
    cm = np.asarray(result['cm'])
    k = len(labels)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    peak = int(cm.max()) if cm.size else 1
    for i in range(k):
        for j in range(k):
            value = int(cm[i, j])
            color = 'white' if peak and value > peak / 2 else 'black'
            ax.text(j, i, str(value), ha='center', va='center', color=color)
    n_ok, n = result['n_correct'], result['n_test']
    ax.set_title(f'{SYNTHETIC_NOTE}\nAccuracy {n_ok}/{n} = {n_ok / n:.3f}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description='Held-out eval on synthetic 4-channel EMG')
    ap.add_argument('--out', default=os.path.join('demos', 'confusion_matrix.png'))
    ap.add_argument('--seed', type=int, default=None, help='default: sensor.EMG_SEED')
    ap.add_argument('--train', type=int, default=None, help='trials per class for training')
    ap.add_argument('--test', type=int, default=None, help='held-out trials per class')
    args = ap.parse_args()
    kwargs = {}
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if args.train is not None:
        kwargs['n_train'] = args.train
    if args.test is not None:
        kwargs['n_test'] = args.test
    result = evaluate(**kwargs)
    print(format_report(result))
    save_confusion_png(result, args.out)
    print(f'Saved confusion matrix to {args.out}')


if __name__ == '__main__':
    main()
