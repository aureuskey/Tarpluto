# Tarpluto - silent-command features and nearest-centroid decoder.
# Inputs are synthetic 4-channel EMG trials from sensor.SyntheticJawEMG.
import numpy as np

from sensor import (
    CHANNEL_NAMES,
    FS,
    N_CHANNELS,
    N_SAMPLES,
    SILENT_COMMANDS,
    SyntheticJawEMG,
    EMG_SEED,
)

# Held-out protocol used by sim/decode_eval.py and by the demo's training fit.
N_TRAIN = 40
N_TEST = 20
WIN_S = 0.20
HOP_S = 0.20
WIN = int(round(WIN_S * FS))
HOP = int(round(HOP_S * FS))
ZC_DEADZONE = 6.0
# Maps synthetic trial RMS into the 0-1 Patch signal. Not a sensor calibration.
SIGNAL_RMS_SCALE = 40.0
SYNTHETIC_NOTE = 'Synthetic-data numbers only. No sensor was read.'
FEATURE_KINDS = ('rms', 'mav', 'zc', 'wl')


def window_starts(n_samples=N_SAMPLES, win=WIN, hop=HOP):
    if win <= 0 or hop <= 0:
        raise ValueError('window and hop must be positive')
    if win > n_samples:
        raise ValueError('window longer than trial')
    return list(range(0, n_samples - win + 1, hop))


def n_windows(n_samples=N_SAMPLES):
    return len(window_starts(n_samples))


def n_features():
    return n_windows() * len(FEATURE_KINDS) * N_CHANNELS


def feature_names():
    names = []
    for w in range(n_windows()):
        for kind in FEATURE_KINDS:
            for ch in CHANNEL_NAMES:
                names.append(f'w{w}_{kind}_{ch}')
    return names


def _rms(seg):
    return np.sqrt(np.mean(seg * seg, axis=0))


def _mav(seg):
    return np.mean(np.abs(seg), axis=0)


def _zero_crossings(seg, deadzone):
    prev, curr = seg[:-1], seg[1:]
    cross = (prev * curr < 0.0) & (np.abs(curr - prev) >= deadzone)
    return cross.sum(axis=0).astype(float)


def _waveform_length(seg):
    return np.abs(np.diff(seg, axis=0)).sum(axis=0)


def featurize(trial):
    """Windowed RMS, MAV, zero crossings and waveform length per channel.

    Each window is mean-centered first so a slow baseline drift does not
    become the feature. Order is window, then (rms, mav, zc, wl), then channel.
    """
    x = np.asarray(trial, dtype=float)
    if x.shape != (N_SAMPLES, N_CHANNELS):
        raise ValueError(f'expected trial shape {(N_SAMPLES, N_CHANNELS)}, got {x.shape}')
    parts = []
    for start in window_starts(x.shape[0]):
        seg = x[start:start + WIN]
        seg = seg - seg.mean(axis=0, keepdims=True)
        parts.extend((_rms(seg), _mav(seg), _zero_crossings(seg, ZC_DEADZONE), _waveform_length(seg)))
    return np.concatenate(parts)


def synthetic_rms_signal(trial):
    """0-1 signal for the Living Node, from synthetic trial RMS."""
    rms = float(np.sqrt(np.mean(np.asarray(trial, dtype=float) ** 2)))
    return float(np.clip(rms / SIGNAL_RMS_SCALE, 0.0, 1.0))


def confirm_act(answer):
    """True only when the reply is y. Anything else leaves the command unacted."""
    return str(answer).strip().lower() == 'y'


class NearestCentroid:
    """Class mean in z-scored feature space. Predicts the nearest mean."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError('X must be 2-D')
        if len(X) != len(y) or len(X) == 0:
            raise ValueError('X and y must be the same non-zero length')
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        Xs = (X - self.mean_) / self.scale_
        seen = list(dict.fromkeys(y.tolist()))
        known = [c for c in SILENT_COMMANDS if c in seen]
        extra = [c for c in seen if c not in SILENT_COMMANDS]
        classes, cents = [], []
        for label in known + extra:
            classes.append(label)
            cents.append(Xs[y == label].mean(axis=0))
        self.classes_ = np.asarray(classes)
        self.centroids_ = np.vstack(cents)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self.mean_) / self.scale_
        dist = np.linalg.norm(Xs[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dist, axis=1)]


def fit_trials(trials, labels):
    X = np.vstack([featurize(t) for t in trials])
    return NearestCentroid().fit(X, labels)


def build_decoder(n_per_class=N_TRAIN, seed=EMG_SEED):
    """Fit on the leading synthetic trials of `seed` (the eval training split)."""
    trials, labels = SyntheticJawEMG(seed=seed).collect(n_per_class)
    return fit_trials(trials, labels)


def confusion_matrix(y_true, y_pred, labels):
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true not in index or pred not in index:
            raise ValueError(f'label outside {labels}: {true!r} -> {pred!r}')
        cm[index[true], index[pred]] += 1
    return cm


def evaluate(n_train=N_TRAIN, n_test=N_TEST, seed=EMG_SEED):
    """Train on the first n_train trials/class, test on the next n_test.

    Both splits are later draws from the same seeded generator, in command
    order. The test trials are not in the training set.
    """
    emg = SyntheticJawEMG(seed=seed)
    train_trials, train_labels = emg.collect(n_train)
    test_trials, test_labels = emg.collect(n_test)
    clf = fit_trials(train_trials, train_labels)
    pred = clf.predict(np.vstack([featurize(t) for t in test_trials]))
    labels = list(SILENT_COMMANDS)
    cm = confusion_matrix(test_labels, pred, labels)
    n_correct = int(np.sum(pred == np.asarray(test_labels)))
    return {
        'n_correct': n_correct,
        'n_test': len(test_labels),
        'n_train': n_train,
        'n_test_per_class': n_test,
        'seed': seed,
        'labels': labels,
        'cm': cm,
        'y_true': list(test_labels),
        'y_pred': [str(p) for p in pred],
        'n_windows': n_windows(),
        'n_channels': N_CHANNELS,
    }


def format_report(result):
    labels = result['labels']
    cm = np.asarray(result['cm'])
    k = int(result['n_correct'])
    n = int(result['n_test'])
    col_w = max(5, max(len(lab) for lab in labels))
    header = ' '.join(f'{lab:>{col_w}}' for lab in labels)
    lines = [
        SYNTHETIC_NOTE,
        'Classifier: nearest centroid',
        (
            f"Features: RMS, MAV, zero crossings, waveform length; "
            f"{result['n_windows']} windows x {result['n_channels']} channels"
        ),
        f"Commands: {', '.join(labels)}",
        (
            f"Train: {result['n_train']} trials/class | "
            f"Test: {result['n_test_per_class']} trials/class | seed {result['seed']}"
        ),
        f'Accuracy: {k}/{n} = {k / n:.3f}',
        '',
        'Confusion matrix (rows = true, columns = predicted):',
        f"{'':>{col_w}} {header}",
    ]
    for i, lab in enumerate(labels):
        cells = ' '.join(f'{int(cm[i, j]):>{col_w}d}' for j in range(len(labels)))
        lines.append(f'{lab:>{col_w}} {cells}')
    return '\n'.join(lines)
