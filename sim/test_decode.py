# Plain tests for the synthetic decoder. From the repo root:
#   python3 sim/test_decode.py
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from decode import (
    N_TEST,
    N_TRAIN,
    SYNTHETIC_NOTE,
    build_decoder,
    confirm_act,
    evaluate,
    featurize,
    feature_names,
    format_report,
    n_features,
    window_starts,
)
from sensor import (
    CHANNEL_NAMES,
    EMG_SEED,
    N_CHANNELS,
    N_SAMPLES,
    ONLINE_SEED,
    SILENT_COMMANDS,
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
    def test_docs_say_v010_and_not_old_versions(self):
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
        self.assertTrue(core.startswith('# Tarpluto - core simulation loop v0.10\n'))
        self.assertIn('VERSION = "v0.10"', core)
        self.assertIn('Tarpluto {VERSION}', core)
        self.assertIn('v0.10', readme)
        self.assertIn('v0.10', roadmap)
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
        self.assertIn('=== Tarpluto v0.10 ===', proc.stdout)
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
            self.assertIn(f'Saved confusion matrix to {out}', proc.stdout)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(8), b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()
