# Plain tests for the synthetic decoder. From the repo root:
#   python3 sim/test_decode.py
import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

from decode import (
    N_RECAL,
    N_TEST,
    N_TRAIN,
    REJECT_LABEL,
    REJECT_MARGIN_PERCENTILE,
    SYNTHETIC_NOTE,
    _split_recal_prefix,
    build_decoder,
    confirm_act,
    evaluate,
    evaluate_rest,
    evaluate_sessions,
    featurize,
    feature_names,
    format_report,
    format_rest_report,
    format_session_report,
    n_features,
    session_effects,
    session_generator_seed,
    window_starts,
)
from sensor import (
    CHANNEL_NAMES,
    EMG_SEED,
    N_CHANNELS,
    N_SAMPLES,
    ONLINE_SEED,
    REST_TWITCH_AMP_HI,
    SILENT_COMMANDS,
    TEMPLATES,
    SessionEffects,
    SyntheticJawEMG,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GeneratorTests(unittest.TestCase):
    def test_shape_and_names(self):
        trial = SyntheticJawEMG(seed=EMG_SEED).trial('yes')
        self.assertEqual(trial.shape, (N_SAMPLES, N_CHANNELS))
        self.assertEqual(len(CHANNEL_NAMES), N_CHANNELS)
        self.assertEqual(SILENT_COMMANDS, ('yes', 'no', 'help', 'water', 'stop'))

    def test_seed_repeats_and_trials_vary(self):
        a = SyntheticJawEMG(seed=3)
        b = SyntheticJawEMG(seed=3)
        first = a.trial('no')
        self.assertTrue(np.allclose(first, b.trial('no')))
        self.assertFalse(np.allclose(first, a.trial('no')))

    def test_commands_are_not_the_same_pattern(self):
        emg = SyntheticJawEMG(seed=5)
        yes = np.mean([emg.trial('yes') for _ in range(8)], axis=0)
        stop = np.mean([emg.trial('stop') for _ in range(8)], axis=0)
        self.assertFalse(np.allclose(yes, stop, atol=1.0))

    def test_unknown_command(self):
        with self.assertRaises(ValueError):
            SyntheticJawEMG(seed=1).trial('maybe')

    def test_online_seed_is_a_different_stream(self):
        train = SyntheticJawEMG(seed=EMG_SEED).trial('yes')
        online = SyntheticJawEMG(seed=ONLINE_SEED).trial('yes')
        self.assertFalse(np.allclose(train, online))


class FeatureTests(unittest.TestCase):
    def test_windows_cover_the_trial(self):
        starts = window_starts()
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + int(round(0.20 * 200)), N_SAMPLES)
        self.assertGreaterEqual(len(starts), 2)

    def test_feature_vector(self):
        trial = SyntheticJawEMG(seed=1).trial('help')
        vec = featurize(trial)
        self.assertEqual(vec.shape, (n_features(),))
        self.assertEqual(len(feature_names()), n_features())
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            featurize(np.zeros((10, N_CHANNELS)))


class ClassifierTests(unittest.TestCase):
    def test_nearest_centroid_on_two_blobs(self):
        from decode import NearestCentroid
        X = np.array([[0.0, 0.0], [0.2, -0.1], [5.0, 5.0], [4.8, 5.2]])
        y = np.array(['a', 'a', 'b', 'b'])
        pred = NearestCentroid().fit(X, y).predict([[0.0, 0.1], [5.1, 4.9]])
        self.assertEqual(list(pred), ['a', 'b'])

    def test_confirm_only_y(self):
        self.assertTrue(confirm_act('y'))
        self.assertTrue(confirm_act('Y'))
        self.assertTrue(confirm_act(' y '))
        for answer in ('', 'n', 'yes', 'yy', ' y n'):
            self.assertFalse(confirm_act(answer))

    def test_held_out_beats_chance_and_is_repeatable(self):
        first = evaluate()
        second = evaluate()
        self.assertEqual(first['n_test'], len(SILENT_COMMANDS) * N_TEST)
        self.assertEqual(first['n_train'], N_TRAIN)
        self.assertEqual(first['cm'].shape, (len(SILENT_COMMANDS), len(SILENT_COMMANDS)))
        self.assertTrue(np.array_equal(first['cm'], second['cm']))
        self.assertEqual(int(np.trace(first['cm'])), first['n_correct'])
        self.assertTrue(np.all(first['cm'].sum(axis=1) == N_TEST))
        # Five labels: chance is 0.2. The synthetic patterns are meant to separate.
        self.assertGreater(first['n_correct'] / first['n_test'], 0.5)
        report = format_report(first)
        self.assertIn(SYNTHETIC_NOTE, report)
        self.assertIn(f"Accuracy: {first['n_correct']}/{first['n_test']}", report)

    def test_update_centroids_keeps_scaler_and_can_change_the_label(self):
        from decode import NearestCentroid
        X = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
        ])
        y = np.array(['a', 'a', 'b', 'b'])
        clf = NearestCentroid().fit(X, y)
        point = np.array([[6.5, 0.0]])
        self.assertEqual(str(clf.predict(point)[0]), 'b')
        fresh = clf.copy()
        mean, scale = clf.mean_.copy(), clf.scale_.copy()
        fresh.update_centroids(
            np.array([[6.0, 0.0], [6.0, 0.0], [10.0, 0.0], [10.0, 0.0]]),
            np.array(['a', 'a', 'b', 'b']),
        )
        self.assertEqual(str(fresh.predict(point)[0]), 'a')
        self.assertEqual(str(clf.predict(point)[0]), 'b')
        self.assertTrue(np.array_equal(fresh.mean_, mean))
        self.assertTrue(np.array_equal(fresh.scale_, scale))
        with self.assertRaises(ValueError):
            clf.copy().update_centroids(np.array([[6.0, 0.0]]), np.array(['a']))

    def test_demo_decoder_matches_eval_training_seed(self):
        result = evaluate(n_train=N_TRAIN, n_test=1)
        demo = build_decoder()
        from decode import featurize as feat
        trial = SyntheticJawEMG(seed=EMG_SEED)
        train, labels = trial.collect(N_TRAIN)
        # One more draw on that stream is the first held-out trial (class "yes").
        held = trial.trial('yes')
        pred = str(demo.predict(feat(held)[None, :])[0])
        self.assertEqual(pred, result['y_pred'][0])
        self.assertEqual(labels[0], 'yes')
        self.assertEqual(len(train), N_TRAIN * len(SILENT_COMMANDS))


class VersionTests(unittest.TestCase):
    def test_docs_say_v012_and_not_old_versions(self):
        paths = [
            os.path.join(ROOT, 'sim', 'core_loop.py'),
            os.path.join(ROOT, 'README.md'),
            os.path.join(ROOT, 'docs', 'roadmap.md'),
            os.path.join(ROOT, 'CHANGELOG.md'),
        ]
        texts = []
        for path in paths:
            with open(path, encoding='utf-8') as f:
                texts.append(f.read())
        for text in texts:
            self.assertNotIn('v0.4', text)
            self.assertNotIn('v0.8', text)
        core, readme, roadmap, changelog = texts
        self.assertTrue(core.startswith('# Tarpluto - core simulation loop v0.12\n'))
        self.assertIn('VERSION = "v0.12"', core)
        self.assertNotIn('v0.11', core)
        self.assertNotIn('v0.10', core)
        self.assertIn('Tarpluto {VERSION}', core)
        self.assertIn('v0.12', readme)
        self.assertIn('v0.12', roadmap)
        self.assertIn('synthetic data, simulated sessions', readme.lower())
        self.assertIn('synthetic data, simulated sessions', roadmap.lower())
        self.assertIn('## v0.12', changelog)
        self.assertIn('## v0.11', changelog)
        self.assertIn('synthetic data, simulated sessions', changelog.lower())
        self.assertIn(SYNTHETIC_NOTE, readme)
        self.assertIn(SYNTHETIC_NOTE, roadmap)
        self.assertIn(SYNTHETIC_NOTE, changelog)


def _run_loop(args, stdin, figure):
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, 'sim', 'core_loop.py'), *args, '--figure', figure],
        input=stdin,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        timeout=60,
    )


class LoopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fig = os.path.join(self._tmp.name, 'fig.png')

    def tearDown(self):
        self._tmp.cleanup()

    def test_demo_still_runs_without_decoding(self):
        proc = _run_loop(['--demo'], 'quit\n', self.fig)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('=== Tarpluto v0.12 ===', proc.stdout)
        self.assertIn('Type a thought. Take longer to think', proc.stdout)
        self.assertNotIn('[Decode]', proc.stdout)
        self.assertNotIn('Type y to confirm', proc.stdout)

    def test_decode_acts_only_after_y(self):
        refused = _run_loop(['--demo', '--decode'], 'yes\nn\nstatus\nquit\n', self.fig)
        self.assertEqual(refused.returncode, 0, refused.stderr)
        self.assertIn(SYNTHETIC_NOTE, refused.stdout)
        self.assertIn("synthesized 'yes'", refused.stdout)
        self.assertIn('Type y to confirm', refused.stdout)
        self.assertIn('Not confirmed. No command acted.', refused.stdout)
        self.assertIn('thoughts 0', refused.stdout)
        self.assertNotIn('[Living Node]', refused.stdout)

        accepted = _run_loop(['--demo', '--decode'], 'yes\ny\nstatus\nquit\n', self.fig)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn('[Neural Patch]', accepted.stdout)
        self.assertIn('[Living Node]', accepted.stdout)
        self.assertIn('thoughts 1', accepted.stdout)
        self.assertIn('[AI]', accepted.stdout)

    def test_decode_rest_is_none_and_does_not_act(self):
        proc = _run_loop(['--demo', '--decode'], 'rest\nstatus\nquit\n', self.fig)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("synthesized 'rest' | predicted 'none'", proc.stdout)
        self.assertIn('Rejected as none. No command acted.', proc.stdout)
        self.assertIn('thoughts 0', proc.stdout)
        self.assertNotIn('[Living Node]', proc.stdout)
        self.assertNotIn('Type y to confirm', proc.stdout)

    def test_fake4ch_is_the_decoder(self):
        proc = _run_loop(['--source', 'fake4ch'], 'stop\ny\nquit\n', self.fig)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("synthesized 'stop'", proc.stdout)
        self.assertIn('[Living Node]', proc.stdout)
        self.assertNotIn('[Calibrate]', proc.stdout)

    def test_decode_rejects_serial(self):
        proc = _run_loop(
            ['--decode', '--source', 'serial', '--port', '/dev/null'],
            'quit\n',
            self.fig,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('synthetic 4-channel', proc.stderr)


class EvalScriptTests(unittest.TestCase):
    def test_script_prints_report_and_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'cm.png')
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, 'sim', 'decode_eval.py'), '--out', out],
                text=True,
                capture_output=True,
                cwd=ROOT,
                env=env,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = format_report(evaluate())
            self.assertIn(report, proc.stdout)
            self.assertNotIn('False activation rate on rest', proc.stdout)
            self.assertIn(f'Saved confusion matrix to {out}', proc.stdout)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')

    def test_sessions_script_prints_report_and_small_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'sessions.png')
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            proc = subprocess.run(
                [
                    sys.executable, os.path.join(ROOT, 'sim', 'decode_eval.py'),
                    '--sessions', '3', '--out', out,
                ],
                text=True,
                capture_output=True,
                cwd=ROOT,
                env=env,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = format_session_report(evaluate_sessions(n_sessions=3))
            self.assertIn(report, proc.stdout)
            self.assertIn('synthetic data, simulated sessions', proc.stdout)
            self.assertNotIn('False activation rate on rest', proc.stdout)
            self.assertIn(f'Saved session accuracy plot to {out}', proc.stdout)
            self.assertLess(os.path.getsize(out), 200_000)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')

    def test_rest_script_prints_report_and_small_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'rest.png')
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, 'sim', 'decode_eval.py'), '--rest', '--out', out],
                text=True,
                capture_output=True,
                cwd=ROOT,
                env=env,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = format_rest_report(evaluate_rest())
            self.assertIn(report, proc.stdout)
            self.assertIn(SYNTHETIC_NOTE, proc.stdout)
            self.assertIn(f'Saved rest rejection plot to {out}', proc.stdout)
            self.assertLess(os.path.getsize(out), 200_000)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')

    def test_rest_and_sessions_are_not_combined(self):
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        proc = subprocess.run(
            [
                sys.executable, os.path.join(ROOT, 'sim', 'decode_eval.py'),
                '--rest', '--sessions', '2',
            ],
            text=True,
            capture_output=True,
            cwd=ROOT,
            env=env,
            timeout=60,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('separate evals', proc.stderr)


class PerturbationTests(unittest.TestCase):
    def test_zero_session_matches_the_plain_generator(self):
        plain = SyntheticJawEMG(seed=EMG_SEED)
        wrapped = SyntheticJawEMG(seed=EMG_SEED, session=SessionEffects(seed=123))
        left, left_y = plain.collect(3)
        right, right_y = wrapped.collect(3)
        self.assertEqual(left_y, right_y)
        for a, b in zip(left, right):
            self.assertTrue(np.array_equal(a, b))

    def test_electrode_shift_remixes_and_partially_mixes(self):
        effects = SessionEffects(seed=5, electrode_shift=0.7)
        off_diag = ~np.eye(N_CHANNELS, dtype=bool)
        self.assertTrue(np.all(effects.mix[off_diag] > 0.0))
        for i in range(N_CHANNELS):
            column = np.delete(effects.mix[:, i], i)
            self.assertGreater(float(np.max(column) - np.min(column)), 1e-6)
        hot = np.zeros((N_SAMPLES, N_CHANNELS))
        hot[:, 0] = 1.0
        mixed = effects.apply(hot, trial_index=0, n_trials=1)
        self.assertGreater(float(np.sum(mixed[:, 1:] ** 2)), 0.05)
        again = SessionEffects(seed=5, electrode_shift=0.7).apply(hot, 0, 1)
        self.assertTrue(np.allclose(mixed, again))
        self.assertFalse(np.allclose(
            effects.mix, SessionEffects(seed=6, electrode_shift=0.7).mix,
        ))

    def test_amplitude_drift_is_a_fixed_per_channel_gain(self):
        effects = SessionEffects(seed=8, amplitude_drift=0.25)
        self.assertFalse(np.allclose(effects.gains, np.ones(N_CHANNELS)))
        self.assertGreater(float(np.std(effects.gains)), 0.0)
        ones = np.ones((12, N_CHANNELS))
        self.assertTrue(np.allclose(effects.apply(ones, 0, 1), effects.gains))
        hot = np.zeros((12, N_CHANNELS))
        hot[:, 2] = 3.0
        scaled = effects.apply(hot, 0, 1)
        self.assertTrue(np.allclose(scaled[:, 2], 3.0 * effects.gains[2]))
        self.assertTrue(np.allclose(scaled[:, [0, 1, 3]], 0.0))
        other = SessionEffects(seed=9, amplitude_drift=0.25).gains
        self.assertFalse(np.allclose(effects.gains, other))

    def test_fatigue_drop_scales_later_trials_only(self):
        drop = 0.4
        plain = SyntheticJawEMG(seed=9)
        pert = SyntheticJawEMG(seed=9, session=SessionEffects(seed=1, fatigue_drop=drop))
        base, _ = plain.collect(5)
        got, _ = pert.collect(5)
        self.assertTrue(np.array_equal(got[0], base[0]))
        self.assertTrue(np.allclose(got[4], base[4] * (1.0 - drop)))

    def test_fatigue_spectral_changes_zero_crossings(self):
        from decode import ZC_DEADZONE, _zero_crossings
        effects = SessionEffects(seed=2, fatigue_spectral=0.85)
        base = SyntheticJawEMG(seed=4).trial('stop')
        early = effects.apply(base, trial_index=0, n_trials=10)
        late = effects.apply(base, trial_index=9, n_trials=10)
        self.assertTrue(np.array_equal(early, base))
        zc = lambda trial: float(_zero_crossings(trial, ZC_DEADZONE).sum())
        self.assertGreater(zc(early), zc(late))
        scale = float(np.dot(late.ravel(), early.ravel()) / np.dot(early.ravel(), early.ravel()))
        residual = late - scale * early
        self.assertGreater(float(np.sqrt(np.mean(residual ** 2))), 1.0)

    def test_rejects_bad_strengths(self):
        with self.assertRaises(ValueError):
            SessionEffects(seed=1, electrode_shift=1.2)
        with self.assertRaises(ValueError):
            SessionEffects(seed=1, amplitude_drift=-0.1)
        with self.assertRaises(ValueError):
            SessionEffects(seed=1, fatigue_drop=-0.1)
        with self.assertRaises(ValueError):
            SessionEffects(seed=1, fatigue_spectral=1.5)


class SessionEvalTests(unittest.TestCase):
    def test_recal_prefix_is_held_out_of_the_test(self):
        trials = list(range(25))
        labels = [c for c in SILENT_COMMANDS for _ in range(5)]
        recal_t, recal_y, test_t, test_y = _split_recal_prefix(trials, labels, 2, 3)
        self.assertEqual(recal_y, [c for c in SILENT_COMMANDS for _ in range(2)])
        self.assertEqual(test_y, [c for c in SILENT_COMMANDS for _ in range(3)])
        self.assertEqual(recal_t, [0, 1, 5, 6, 10, 11, 15, 16, 20, 21])
        self.assertEqual(test_t, [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24])

    def test_later_session_applies_effects_on_the_same_base_draw(self):
        seed = session_generator_seed(2)
        plain = SyntheticJawEMG(seed=seed)
        effects = session_effects(2)
        pert = SyntheticJawEMG(seed=seed, session=session_effects(2))
        base, _ = plain.collect(4)
        got, _ = pert.collect(4)
        self.assertFalse(np.allclose(base[0], got[0]))
        expected = effects.apply(base[3], trial_index=3, n_trials=4)
        self.assertTrue(np.allclose(got[3], expected))

    def test_session_eval_is_repeatable_and_session1_matches_held_out(self):
        from core_loop import VERSION
        first = evaluate_sessions(n_sessions=3)
        second = evaluate_sessions(n_sessions=3)
        self.assertEqual(first['rows'], second['rows'])
        held = evaluate()
        row = first['rows'][0]
        self.assertEqual(row['session'], 1)
        self.assertFalse(row['perturbed'])
        self.assertEqual(row['seed'], EMG_SEED)
        self.assertEqual(row['no_recal_correct'], held['n_correct'])
        self.assertEqual(row['n_test'], held['n_test'])
        self.assertEqual(first['n_recal'], N_RECAL)
        self.assertEqual(len(first['rows']), 3)
        for later in first['rows'][1:]:
            self.assertTrue(later['perturbed'])
            self.assertEqual(later['n_test'], held['n_test'])
            self.assertGreaterEqual(later['no_recal_correct'], 0)
            self.assertLessEqual(later['no_recal_correct'], later['n_test'])
            self.assertGreaterEqual(later['recal_correct'], 0)
            self.assertLessEqual(later['recal_correct'], later['n_test'])
        report = format_session_report(first)
        self.assertIn('synthetic data, simulated sessions', report)
        self.assertIn(SYNTHETIC_NOTE, report)
        self.assertIn(f'recal ({N_RECAL}/class)', report)
        self.assertIn(f'Tarpluto {VERSION} cross-session robustness', report)
        self.assertIn('3 labelled trials/class', report)

    def test_rejects_empty_protocol(self):
        with self.assertRaises(ValueError):
            evaluate_sessions(n_sessions=0)
        with self.assertRaises(ValueError):
            evaluate_sessions(n_recal=0)


def _rms(trial):
    return float(np.sqrt(np.mean(np.asarray(trial, dtype=float) ** 2)))


class RestGeneratorTests(unittest.TestCase):
    def test_shape_seed_and_finite(self):
        first = SyntheticJawEMG(seed=11).collect_rest(4)
        second = SyntheticJawEMG(seed=11).collect_rest(4)
        self.assertEqual(len(first), 4)
        for a, b in zip(first, second):
            self.assertEqual(a.shape, (N_SAMPLES, N_CHANNELS))
            self.assertTrue(np.all(np.isfinite(a)))
            self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_fixed_seed_ignores_global_numpy_seed(self):
        np.random.seed(0)
        left = SyntheticJawEMG(seed=5).rest()
        np.random.seed(1)
        right = SyntheticJawEMG(seed=5).rest()
        self.assertTrue(np.array_equal(left, right))

    def test_rest_is_not_a_command_template(self):
        for name in ('rest', 'none', 'idle'):
            with self.assertRaises(ValueError):
                SyntheticJawEMG(seed=1).trial(name)

    def test_calling_trial_only_does_not_draw_rest(self):
        plain = SyntheticJawEMG(seed=9)
        other = SyntheticJawEMG(seed=9)
        self.assertTrue(np.array_equal(plain.trial('yes'), other.trial('yes')))
        self.assertTrue(np.array_equal(plain.trial('stop'), other.trial('stop')))
        mixed = SyntheticJawEMG(seed=9)
        mixed.rest()
        self.assertFalse(np.array_equal(mixed.trial('yes'), SyntheticJawEMG(seed=9).trial('yes')))

    def test_rest_is_quieter_than_commands_and_not_flat(self):
        commands, _ = SyntheticJawEMG(seed=EMG_SEED).collect(4)
        rest = SyntheticJawEMG(seed=EMG_SEED).collect_rest(40)
        cmd_rms = np.array([_rms(t) for t in commands])
        rest_rms = np.array([_rms(t) for t in rest])
        self.assertLess(float(rest_rms.mean()), float(cmd_rms.mean()) * 0.5)
        self.assertLess(float(rest_rms.max()), float(cmd_rms.min()))
        self.assertGreater(float(rest_rms.max() - rest_rms.min()), 0.5)
        min_command_amp = min(amp for bumps in TEMPLATES.values() for *_, amp in bumps)
        self.assertLess(REST_TWITCH_AMP_HI, min_command_amp)

    def test_collect_rest_rejects_empty(self):
        with self.assertRaises(ValueError):
            SyntheticJawEMG(seed=1).collect_rest(0)


class RejectionTests(unittest.TestCase):
    def test_unset_threshold_still_labels_the_midpoint(self):
        from decode import NearestCentroid
        X = np.array([
            [0.0, 0.0],
            [0.2, -0.1],
            [0.1, 0.2],
            [8.0, 0.0],
            [8.1, 0.2],
            [7.9, -0.1],
        ])
        y = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
        clf = NearestCentroid().fit(X, y)
        self.assertIsNone(clf.threshold_)
        mid = np.array([[4.0, 0.0]])
        self.assertIn(str(clf.predict(mid)[0]), ('a', 'b'))
        self.assertNotEqual(str(clf.predict(mid)[0]), REJECT_LABEL)

    def test_margin_gate_keeps_a_centroid_and_rejects_the_middle(self):
        from decode import NearestCentroid
        X = np.array([
            [0.0, 0.0],
            [0.2, -0.1],
            [0.1, 0.2],
            [8.0, 0.0],
            [8.1, 0.2],
            [7.9, -0.1],
        ])
        y = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
        clf = NearestCentroid().fit(X, y)
        mid = np.array([[4.0, 0.0]])
        home = np.array([[0.0, 0.0]])
        m_mid, _ = clf.nearest_margin(mid)
        m_home, lab_home = clf.nearest_margin(home)
        self.assertEqual(str(lab_home[0]), 'a')
        self.assertLess(float(m_mid[0]), float(m_home[0]))
        clf.threshold_ = float((m_mid[0] + m_home[0]) / 2.0)
        self.assertEqual(str(clf.predict(home)[0]), 'a')
        self.assertEqual(str(clf.predict(mid)[0]), REJECT_LABEL)
        copied = clf.copy()
        self.assertEqual(copied.threshold_, clf.threshold_)
        self.assertEqual(str(copied.predict(mid)[0]), REJECT_LABEL)
        fresh = NearestCentroid().fit(X, y)
        self.assertIsNone(fresh.threshold_)
        self.assertNotEqual(str(fresh.predict(mid)[0]), REJECT_LABEL)

    def test_threshold_is_the_training_percentile_only(self):
        from decode import NearestCentroid
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [6.0, 0.0],
            [6.1, 0.0],
            [6.0, 0.2],
        ], dtype=float)
        y = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
        clf = NearestCentroid().fit(X, y)
        before = clf.centroids_.copy()
        clf.calibrate_rejection(X, percentile=1.0)
        margin, _ = clf.nearest_margin(X)
        expected = float(np.percentile(margin, 1.0, method='linear'))
        self.assertEqual(clf.threshold_, expected)
        self.assertEqual(clf.reject_percentile_, 1.0)
        self.assertTrue(np.array_equal(clf.centroids_, before))
        # A second calibration on the same rows does not look at other points.
        other = np.array([[100.0, 100.0], [-100.0, 50.0]])
        again = NearestCentroid().fit(X, y).calibrate_rejection(X, percentile=1.0)
        self.assertEqual(again.threshold_, clf.threshold_)
        self.assertNotEqual(
            NearestCentroid().fit(np.vstack([X, other]), np.array(['a', 'a', 'a', 'b', 'b', 'b', 'a', 'b'])).calibrate_rejection(
                np.vstack([X, other]), percentile=1.0,
            ).threshold_,
            clf.threshold_,
        )

    def test_calibrate_rejects_bad_input(self):
        from decode import NearestCentroid
        clf = NearestCentroid()
        with self.assertRaises(ValueError):
            clf.calibrate_rejection(np.zeros((2, 2)))
        X = np.array([[0.0, 0.0], [1.0, 1.0], [8.0, 0.0], [9.0, 0.1]])
        y = np.array(['a', 'a', 'b', 'b'])
        fitted = NearestCentroid().fit(X, y)
        with self.assertRaises(ValueError):
            fitted.calibrate_rejection(X, percentile=101)
        with self.assertRaises(ValueError):
            fitted.calibrate_rejection(np.zeros((0, 2)))


class RestEvalTests(unittest.TestCase):
    def test_rest_eval_matches_open_decoder_and_accounts(self):
        first = evaluate_rest()
        second = evaluate_rest()
        held = evaluate()
        self.assertEqual(first['y_true'], second['y_true'])
        self.assertEqual(first['y_pred'], second['y_pred'])
        self.assertEqual(first['y_pred_open'], held['y_pred'])
        self.assertEqual(first['y_true'], held['y_true'])
        self.assertEqual(first['threshold'], second['threshold'])
        n = first['n_command']
        self.assertEqual(
            first['n_correct'] + first['n_rejected'] + first['n_wrong_command'],
            n,
        )
        self.assertEqual(first['n_false_activation'] + first['n_rest_rejected'], first['n_rest'])
        self.assertEqual(first['n_rest'], n)
        self.assertEqual(first['n_command'], held['n_test'])
        self.assertGreaterEqual(first['n_correct'] / n, 0.9)
        self.assertLessEqual(first['n_rejected'] / n, 0.1)
        self.assertEqual(first['n_false_activation'], 0)
        self.assertEqual(first['percentile'], REJECT_MARGIN_PERCENTILE)
        op = [row for row in first['sweep'] if row['percentile'] == first['percentile']]
        self.assertEqual(len(op), 1)
        self.assertEqual(op[0]['threshold'], first['threshold'])
        self.assertEqual(op[0]['n_correct'], first['n_correct'])
        self.assertEqual(op[0]['n_rejected'], first['n_rejected'])
        self.assertEqual(op[0]['n_false_activation'], first['n_false_activation'])
        # The gate is the 1st percentile of the training margins, not the test set.
        emg = SyntheticJawEMG(seed=EMG_SEED)
        train, _ = emg.collect(N_TRAIN)
        from decode import fit_trials
        clf = fit_trials(train, _)
        X = np.vstack([featurize(t) for t in train])
        margin, _ = clf.nearest_margin(X)
        expected = float(np.percentile(margin, REJECT_MARGIN_PERCENTILE, method='linear'))
        self.assertEqual(first['threshold'], expected)
        report = format_rest_report(first)
        self.assertIn(SYNTHETIC_NOTE, report)
        self.assertIn('training commands only', report)
        self.assertIn(f"Command accuracy with rejection: {first['n_correct']}/{n}", report)
        self.assertIn(
            f"Wrong-rejection rate (command windows predicted none): {first['n_rejected']}/{n}",
            report,
        )
        self.assertIn(
            f"False activation rate on rest: {first['n_false_activation']}/{first['n_rest']}",
            report,
        )
        self.assertIn(f"Threshold: {first['threshold']:.6f}", report)
        self.assertIn('1st percentile', report)

    def test_online_yes_is_accepted_and_online_rest_is_none(self):
        clf = build_decoder(reject=True)
        self.assertIsNone(build_decoder().threshold_)
        yes = SyntheticJawEMG(seed=ONLINE_SEED).trial('yes')
        idle = SyntheticJawEMG(seed=ONLINE_SEED).rest()
        self.assertEqual(str(clf.predict(featurize(yes)[None, :])[0]), 'yes')
        self.assertEqual(str(clf.predict(featurize(idle)[None, :])[0]), REJECT_LABEL)
        self.assertEqual(clf.threshold_, evaluate_rest()['threshold'])

    def test_rejects_empty_rest_protocol(self):
        with self.assertRaises(ValueError):
            evaluate_rest(n_rest=0)


if __name__ == '__main__':
    unittest.main()
