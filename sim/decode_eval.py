# Held-out eval for the synthetic silent-command decoder.
# Run from the repo root: python3 sim/decode_eval.py
# Cross-session: python3 sim/decode_eval.py --sessions 4
# Re-run and replace demos/confusion_matrix.png if the generator or features change.
# Re-run and replace demos/session_accuracy.png if the session protocol changes.
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core_loop import VERSION
from decode import (
    SYNTHETIC_NOTE,
    evaluate,
    evaluate_sessions,
    format_report,
    format_session_report,
)


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


def save_session_png(result, path):
    """Small accuracy-by-session plot. Synthetic data, simulated sessions."""
    rows = result['rows']
    sessions = [row['session'] for row in rows]
    norecal = [row['no_recal_correct'] / row['n_test'] for row in rows]
    recal = [row['recal_correct'] / row['n_test'] for row in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(sessions, norecal, 'o-', label='no recalibration')
    ax.plot(sessions, recal, 's-', label=f"recal ({result['n_recal']}/class)")
    ax.set_ylim(-0.02, 1.05)
    pad = 0.2
    ax.set_xlim(min(sessions) - pad, max(sessions) + pad)
    ax.set_xticks(sessions)
    ax.set_xlabel('Simulated session')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Tarpluto {VERSION}\nsynthetic data, simulated sessions')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description='Held-out and cross-session eval on synthetic 4-channel EMG',
    )
    ap.add_argument('--out', default=None, help='png path (default depends on the mode)')
    ap.add_argument('--seed', type=int, default=None, help='default: sensor.EMG_SEED')
    ap.add_argument('--train', type=int, default=None, help='trials per class for training')
    ap.add_argument('--test', type=int, default=None, help='held-out trials per class')
    ap.add_argument('--sessions', type=int, default=None,
                    help='cross-session eval over N simulated sessions (synthetic data)')
    ap.add_argument('--recal', type=int, default=None,
                    help='labelled trials per command used to update centroids (default 3)')
    args = ap.parse_args()
    if args.sessions is not None:
        kwargs = {'n_sessions': args.sessions}
        if args.seed is not None:
            kwargs['seed'] = args.seed
        if args.train is not None:
            kwargs['n_train'] = args.train
        if args.test is not None:
            kwargs['n_test'] = args.test
        if args.recal is not None:
            kwargs['n_recal'] = args.recal
        result = evaluate_sessions(**kwargs)
        print(format_session_report(result))
        out = args.out or os.path.join('demos', 'session_accuracy.png')
        save_session_png(result, out)
        print(f'Saved session accuracy plot to {out}')
        return
    kwargs = {}
    if args.seed is not None:
        kwargs['seed'] = args.seed
    if args.train is not None:
        kwargs['n_train'] = args.train
    if args.test is not None:
        kwargs['n_test'] = args.test
    result = evaluate(**kwargs)
    print(format_report(result))
    out = args.out or os.path.join('demos', 'confusion_matrix.png')
    save_confusion_png(result, out)
    print(f'Saved confusion matrix to {out}')


if __name__ == '__main__':
    main()
