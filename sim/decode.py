# Tarpluto - silent-command features and nearest-centroid decoder.
# Inputs are synthetic 4-channel EMG trials from sensor.SyntheticJawEMG.
import numpy as np

from sensor import (
    CHANNEL_NAMES,
    FS,
    N_CHANNELS,
    N_SAMPLES,
    SILENT_COMMANDS,
    SessionEffects,
    SyntheticJawEMG,
    EMG_SEED,
)

# Cross-session protocol. Fixed simulation knobs, not a fit to a recording.
# Session 1 leaves them off so the same-session split matches evaluate().
N_RECAL = 3
N_SESSIONS = 4
SESSION_ELECTRODE_SHIFT = 0.55
SESSION_AMPLITUDE_DRIFT = 0.30
SESSION_FATIGUE_DROP = 0.45
SESSION_FATIGUE_SPECTRAL = 0.70
SESSION_SEED_STRIDE = 100
PERTURB_SEED_OFFSET = 50_000

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

# Idle rejection. The margin is the second-nearest centroid distance minus
# the nearest, in the training z-scored feature space. The threshold is this
# percentile of the training-command margins. It is a fixed rule: the
# held-out sweep in evaluate_rest does not choose it.
REJECT_LABEL = 'none'
REJECT_MARGIN_PERCENTILE = 1.0
REST_SWEEP_PERCENTILES = (0.0, 1.0, 2.0, 5.0, 10.0)


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
    """Class mean in z-scored feature space. Predicts the nearest mean.

    After calibrate_rejection(), a window whose nearest centroid is not
    clearly nearer than the runner-up is 'none'. The threshold is a
    percentile of training margins, not a fit to held-out rest.
    """

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
        self.threshold_ = None
        self.reject_percentile_ = None
        return self

    def copy(self):
        """A separate model. Updating one copy does not move the other."""
        if not hasattr(self, 'centroids_'):
            raise ValueError('fit the decoder before copying it')
        other = NearestCentroid()
        other.mean_ = self.mean_.copy()
        other.scale_ = self.scale_.copy()
        other.classes_ = self.classes_.copy()
        other.centroids_ = self.centroids_.copy()
        other.threshold_ = self.threshold_
        other.reject_percentile_ = self.reject_percentile_
        return other

    def update_centroids(self, X, y):
        """Replace centroids with the mean of these labelled trials.

        The scaler from fit() stays. Used for a short recalibration.
        """
        if not hasattr(self, 'centroids_'):
            raise ValueError('fit the decoder before updating centroids')
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError('X and y must be the same non-zero length')
        known = set(self.classes_.tolist())
        for lab in y.tolist():
            if lab not in known:
                raise ValueError(f'recalibration label outside fitted classes: {lab!r}')
        Xs = (X - self.mean_) / self.scale_
        cents = []
        for label in self.classes_:
            mask = y == label
            if not np.any(mask):
                raise ValueError(f'no recalibration trials for {label!r}')
            cents.append(Xs[mask].mean(axis=0))
        self.centroids_ = np.vstack(cents)
        return self

    def _distances(self, X):
        Xs = (X - self.mean_) / self.scale_
        return np.linalg.norm(Xs[:, None, :] - self.centroids_[None, :, :], axis=2)

    def nearest_margin(self, X):
        """Margin (second nearest minus nearest) and the nearest label.

        Both are in the z-scored space from fit(). A large margin means one
        command is clearly nearer than the others. Needs two classes.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or len(X) == 0:
            raise ValueError('X must be a non-empty 2-D array')
        if len(self.classes_) < 2:
            raise ValueError('margin needs at least two classes')
        dist = self._distances(X)
        nearest = np.argmin(dist, axis=1)
        part = np.partition(dist, 1, axis=1)
        margin = part[:, 1] - part[:, 0]
        return margin, self.classes_[nearest]

    def calibrate_rejection(self, X, percentile=REJECT_MARGIN_PERCENTILE):
        """Set the margin threshold from these trials only.

        Windows with a smaller margin than this percentile of `X` later
        predict 'none'. `X` is the training commands in the shipped eval.
        Held-out commands and held-out rest are not passed in.
        """
        if not hasattr(self, 'centroids_'):
            raise ValueError('fit the decoder before calibrating rejection')
        percentile = float(percentile)
        if not 0.0 <= percentile <= 100.0:
            raise ValueError('percentile must be between 0 and 100')
        margin, _ = self.nearest_margin(X)
        self.threshold_ = float(np.percentile(margin, percentile, method='linear'))
        self.reject_percentile_ = percentile
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if getattr(self, 'threshold_', None) is None:
            dist = self._distances(X)
            return self.classes_[np.argmin(dist, axis=1)]
        margin, labels = self.nearest_margin(X)
        out = labels.astype(object)
        out[margin < self.threshold_] = REJECT_LABEL
        return out


def fit_trials(trials, labels):
    X = np.vstack([featurize(t) for t in trials])
    return NearestCentroid().fit(X, labels)


def build_decoder(n_per_class=N_TRAIN, seed=EMG_SEED, reject=False,
                  percentile=REJECT_MARGIN_PERCENTILE):
    """Fit on the leading synthetic trials of `seed` (the eval training split).

    reject=False leaves predict() as a plain nearest centroid. reject=True
    sets the margin threshold from those same training trials.
    """
    trials, labels = SyntheticJawEMG(seed=seed).collect(n_per_class)
    clf = fit_trials(trials, labels)
    if reject:
        clf.calibrate_rejection(
            np.vstack([featurize(t) for t in trials]), percentile=percentile,
        )
    return clf


def percentile_phrase(percentile):
    """'1st percentile' for 1, '2nd percentile' for 2, and so on."""
    value = float(percentile)
    if value == int(value):
        n = int(value)
        if 10 <= (n % 100) <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        head = f'{n}{suffix}'
    else:
        head = f'{value:g}th'
    return f'{head} percentile'


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


def _count_rate(k, n):
    return f'{int(k)}/{int(n)} = {k / n:.3f}'


def _rejection_counts(pred, y_true):
    """Correct label, rejected as none, or a different command."""
    n_correct = sum(p == y for p, y in zip(pred, y_true))
    n_rejected = sum(p == REJECT_LABEL for p in pred)
    n_wrong = len(pred) - n_correct - n_rejected
    return n_correct, n_rejected, n_wrong


def evaluate_rest(
    n_train=N_TRAIN,
    n_test=N_TEST,
    n_rest=None,
    seed=EMG_SEED,
    percentile=REJECT_MARGIN_PERCENTILE,
):
    """Held-out commands, then rest windows, from one generator.

    The command split matches evaluate(): n_train then n_test on `seed`.
    Rest windows are drawn after that, so they are not training data.
    The margin threshold is the given percentile of training-command
    margins only. The sweep reprints that rule at a few other percentiles.
    It does not pick the operating point.
    """
    n_train = int(n_train)
    n_test = int(n_test)
    if n_rest is None:
        n_rest = n_test * len(SILENT_COMMANDS)
    n_rest = int(n_rest)
    percentile = float(percentile)
    if n_train < 1 or n_test < 1 or n_rest < 1:
        raise ValueError('n_train, n_test and n_rest must be >= 1')

    emg = SyntheticJawEMG(seed=seed)
    train_trials, train_labels = emg.collect(n_train)
    test_trials, test_labels = emg.collect(n_test)
    rest_trials = emg.collect_rest(n_rest)
    clf = fit_trials(train_trials, train_labels)
    X_train = _featurize_trials(train_trials)
    X_test = _featurize_trials(test_trials)
    X_rest = _featurize_trials(rest_trials)
    y_test = [str(y) for y in test_labels]

    open_pred = [str(p) for p in clf.predict(X_test)]
    clf.calibrate_rejection(X_train, percentile=percentile)
    pred = [str(p) for p in clf.predict(X_test)]
    rest_pred = [str(p) for p in clf.predict(X_rest)]
    n_correct, n_rejected, n_wrong = _rejection_counts(pred, y_test)
    n_false = sum(p != REJECT_LABEL for p in rest_pred)
    n_open = sum(p == y for p, y in zip(open_pred, y_test))

    train_margin, _ = clf.nearest_margin(X_train)
    percents = [float(p) for p in REST_SWEEP_PERCENTILES]
    if percentile not in percents:
        percents.append(percentile)
        percents.sort()
    sweep = []
    saved = clf.threshold_
    try:
        for p in percents:
            clf.threshold_ = float(np.percentile(train_margin, p, method='linear'))
            s_pred = [str(p_) for p_ in clf.predict(X_test)]
            s_rest = [str(p_) for p_ in clf.predict(X_rest)]
            s_correct, s_rejected, _ = _rejection_counts(s_pred, y_test)
            sweep.append({
                'percentile': p,
                'threshold': clf.threshold_,
                'n_correct': s_correct,
                'n_rejected': s_rejected,
                'n_false_activation': sum(p_ != REJECT_LABEL for p_ in s_rest),
                'n_command': len(y_test),
                'n_rest': len(rest_pred),
            })
    finally:
        clf.threshold_ = saved

    return {
        'n_correct': n_correct,
        'n_rejected': n_rejected,
        'n_wrong_command': n_wrong,
        'n_command': len(y_test),
        'n_open_correct': n_open,
        'n_false_activation': n_false,
        'n_rest_rejected': len(rest_pred) - n_false,
        'n_rest': len(rest_pred),
        'n_train': n_train,
        'n_test_per_class': n_test,
        'seed': int(seed),
        'percentile': percentile,
        'threshold': float(clf.threshold_),
        'labels': list(SILENT_COMMANDS),
        'y_true': y_test,
        'y_pred': pred,
        'y_pred_open': open_pred,
        'sweep': sweep,
    }


def format_rest_report(result):
    """Rest-rejection numbers. Synthetic trials. No sensor was read."""
    from core_loop import VERSION
    n = int(result['n_command'])
    m = int(result['n_rest'])
    phrase = percentile_phrase(result['percentile'])
    lines = [
        SYNTHETIC_NOTE,
        f'Tarpluto {VERSION} rest/idle rejection',
        'Each scored window is one synthetic trial (0.80 s).',
        'Classifier: nearest centroid, with a margin gate',
        (
            'Rule: margin = distance to the second-nearest centroid minus '
            'distance to the nearest, in the training z-scored feature space. '
            f"Accept when the margin is at least the {phrase} of "
            f"training-command margins; otherwise output '{REJECT_LABEL}'."
        ),
        (
            'The threshold is computed from the training commands only. '
            'It is not fit on the held-out commands or the held-out rest.'
        ),
        f"Threshold: {result['threshold']:.6f}",
        f"Commands: {', '.join(result['labels'])}",
        (
            f"Train: {result['n_train']} trials/class | "
            f"Test: {result['n_test_per_class']} command trials/class | "
            f"Rest windows: {m} | seed {result['seed']}"
        ),
        (
            'Rest windows are drawn after the command split on that same '
            'generator: sensor noise, 50 Hz hum, baseline drift, occasional '
            'small non-command twitches.'
        ),
        f"Command accuracy without rejection: {_count_rate(result['n_open_correct'], n)}",
        f"Command accuracy with rejection: {_count_rate(result['n_correct'], n)}",
        (
            'Wrong-rejection rate (command windows predicted none): '
            f"{_count_rate(result['n_rejected'], n)}"
        ),
        f"Accepted but wrong command: {_count_rate(result['n_wrong_command'], n)}",
        f"False activation rate on rest: {_count_rate(result['n_false_activation'], m)}",
        '',
        (
            'Threshold sweep (training percentiles). The operating point is '
            f"the {phrase} above, not the best row of this table."
        ),
        (
            f"{'percentile':>10}  {'threshold':>10}  "
            f"{'command_acc':>16}  {'wrong_reject':>16}  {'false_activation':>16}"
        ),
    ]
    for row in result['sweep']:
        lines.append(
            f"{row['percentile']:>10.0f}  {row['threshold']:>10.6f}  "
            f"{_count_rate(row['n_correct'], row['n_command']):>16}  "
            f"{_count_rate(row['n_rejected'], row['n_command']):>16}  "
            f"{_count_rate(row['n_false_activation'], row['n_rest']):>16}"
        )
    return '\n'.join(lines)


def session_generator_seed(session_index, base=EMG_SEED):
    """1-based session index. Session 1 uses `base` (the v0.10 eval seed)."""
    session_index = int(session_index)
    if session_index < 1:
        raise ValueError('session_index is 1-based')
    return int(base) + (session_index - 1) * SESSION_SEED_STRIDE


def session_effects(session_index, base=EMG_SEED,
                    electrode_shift=SESSION_ELECTRODE_SHIFT,
                    amplitude_drift=SESSION_AMPLITUDE_DRIFT,
                    fatigue_drop=SESSION_FATIGUE_DROP,
                    fatigue_spectral=SESSION_FATIGUE_SPECTRAL):
    """Perturbation draw for a later simulated session. Not used on session 1."""
    return SessionEffects(
        seed=session_generator_seed(session_index, base) + PERTURB_SEED_OFFSET,
        electrode_shift=electrode_shift,
        amplitude_drift=amplitude_drift,
        fatigue_drop=fatigue_drop,
        fatigue_spectral=fatigue_spectral,
    )


def _featurize_trials(trials):
    return np.vstack([featurize(t) for t in trials])


def _split_recal_prefix(trials, labels, n_recal, n_test, commands=SILENT_COMMANDS):
    """First n_recal trials of each command are the recalibration set."""
    labels = list(labels)
    block = n_recal + n_test
    recal_t, recal_y, test_t, test_y = [], [], [], []
    for c, command in enumerate(commands):
        start = c * block
        chunk_t = trials[start:start + block]
        chunk_y = labels[start:start + block]
        if len(chunk_t) != block or chunk_y != [command] * block:
            raise RuntimeError('trial block is not grouped by command')
        recal_t.extend(chunk_t[:n_recal])
        recal_y.extend(chunk_y[:n_recal])
        test_t.extend(chunk_t[n_recal:])
        test_y.extend(chunk_y[n_recal:])
    return recal_t, recal_y, test_t, test_y


def _accuracy_pair(clf, test_trials, test_labels, recal_trials, recal_labels):
    """Score the same test trials with the original centroids and an update."""
    X_test = _featurize_trials(test_trials)
    y_test = np.asarray(test_labels)
    no_recal = int(np.sum(clf.predict(X_test) == y_test))
    updated = clf.copy()
    updated.update_centroids(_featurize_trials(recal_trials), recal_labels)
    recal = int(np.sum(updated.predict(X_test) == y_test))
    return no_recal, recal


def evaluate_sessions(
    n_sessions=N_SESSIONS,
    n_train=N_TRAIN,
    n_test=N_TEST,
    n_recal=N_RECAL,
    seed=EMG_SEED,
    electrode_shift=SESSION_ELECTRODE_SHIFT,
    amplitude_drift=SESSION_AMPLITUDE_DRIFT,
    fatigue_drop=SESSION_FATIGUE_DROP,
    fatigue_spectral=SESSION_FATIGUE_SPECTRAL,
):
    """Train on session 1, then score later simulated sessions.

    Session 1 has perturbations off. Its no-recalibration test is the same
    split as evaluate() (n_train then n_test on seed `seed`). Recalibration
    trials for session 1 are drawn after that test.

    Later sessions use electrode shift, per-channel amplitude drift and
    fatigue. The first n_recal trials per command update centroids. The
    next n_test per command are the test, shared by both columns, and sit
    further along the fatigue schedule.
    """
    n_sessions = int(n_sessions)
    n_train = int(n_train)
    n_test = int(n_test)
    n_recal = int(n_recal)
    if n_sessions < 1:
        raise ValueError('n_sessions must be >= 1')
    if n_train < 1 or n_test < 1 or n_recal < 1:
        raise ValueError('n_train, n_test and n_recal must be >= 1')

    emg = SyntheticJawEMG(seed=session_generator_seed(1, seed))
    train_trials, train_labels = emg.collect(n_train)
    clf = fit_trials(train_trials, train_labels)
    test_trials, test_labels = emg.collect(n_test)
    recal_trials, recal_labels = emg.collect(n_recal)
    no_recal, recal = _accuracy_pair(
        clf, test_trials, test_labels, recal_trials, recal_labels,
    )
    rows = [{
        'session': 1,
        'seed': session_generator_seed(1, seed),
        'perturbed': False,
        'no_recal_correct': no_recal,
        'recal_correct': recal,
        'n_test': len(test_labels),
    }]
    for session in range(2, n_sessions + 1):
        effects = session_effects(
            session, seed,
            electrode_shift=electrode_shift,
            amplitude_drift=amplitude_drift,
            fatigue_drop=fatigue_drop,
            fatigue_spectral=fatigue_spectral,
        )
        later = SyntheticJawEMG(
            seed=session_generator_seed(session, seed), session=effects,
        )
        trials, labels = later.collect(n_recal + n_test)
        recal_t, recal_y, test_t, test_y = _split_recal_prefix(
            trials, labels, n_recal, n_test,
        )
        no_recal, recal = _accuracy_pair(clf, test_t, test_y, recal_t, recal_y)
        rows.append({
            'session': session,
            'seed': session_generator_seed(session, seed),
            'perturbed': True,
            'no_recal_correct': no_recal,
            'recal_correct': recal,
            'n_test': len(test_y),
        })
    return {
        'rows': rows,
        'n_sessions': n_sessions,
        'n_train': n_train,
        'n_test_per_class': n_test,
        'n_recal': n_recal,
        'seed': int(seed),
        'labels': list(SILENT_COMMANDS),
        'electrode_shift': float(electrode_shift),
        'amplitude_drift': float(amplitude_drift),
        'fatigue_drop': float(fatigue_drop),
        'fatigue_spectral': float(fatigue_spectral),
        'seed_stride': SESSION_SEED_STRIDE,
        'perturb_seed_offset': PERTURB_SEED_OFFSET,
    }


def format_session_report(result):
    """Per-session accuracies. Synthetic data, simulated sessions."""
    from core_loop import VERSION
    n_recal = int(result['n_recal'])
    labels = result['labels']
    recal_header = f'recal ({n_recal}/class)'
    lines = [
        'synthetic data, simulated sessions',
        SYNTHETIC_NOTE,
        f'Tarpluto {VERSION} cross-session robustness',
        'Classifier: nearest centroid',
        f"Commands: {', '.join(labels)}",
        (
            f"Train: session 1, {result['n_train']} trials/class, "
            f"generator seed {result['seed']}, perturbations off"
        ),
        (
            'Later-session perturbations: '
            f"electrode shift {result['electrode_shift']:.2f} "
            '(channel remix and partial mixing), '
            f"per-channel amplitude drift {result['amplitude_drift']:.2f}, "
            f"fatigue amplitude drop {result['fatigue_drop']:.2f}, "
            f"fatigue spectral blend {result['fatigue_spectral']:.2f}"
        ),
        (
            f"Session k generator seed = {result['seed']} + (k-1)*{result['seed_stride']}. "
            f"Perturbation seed = generator seed + {result['perturb_seed_offset']}."
        ),
        (
            f'Recalibration: {n_recal} labelled trials/class, centroids updated, '
            'session-1 scaler unchanged. Both columns share the same test trials.'
        ),
        (
            'Session 1 recalibration trials are drawn after the held-out test. '
            f"On later sessions the first {n_recal} trials/class recalibrate and "
            f"the next {result['n_test_per_class']} are the test "
            '(later trials carry more fatigue).'
        ),
        f"Test: {result['n_test_per_class']} trials/class per session.",
        '',
        f"{'session':>7} {'seed':>6}  {'perturbation':<22}  "
        f"{'no recalibration':>18}  {recal_header:>18}",
    ]
    for row in result['rows']:
        n = int(row['n_test'])
        a = int(row['no_recal_correct'])
        b = int(row['recal_correct'])
        pert = 'off' if not row['perturbed'] else 'shift, drift, fatigue'
        norecal = f'{a}/{n} = {a / n:.3f}'
        recal = f'{b}/{n} = {b / n:.3f}'
        lines.append(
            f"{row['session']:>7} {row['seed']:>6}  {pert:<22}  "
            f"{norecal:>18}  {recal:>18}"
        )
    return '\n'.join(lines)
