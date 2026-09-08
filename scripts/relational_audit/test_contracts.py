"""Tests for the relational rewrite's parser, pairing, and exact derivations."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import unittest
from collections import Counter
import numpy as np
from analyze_rewrite import CATS, INVALID, parse_answer
from analyze_prompt_matched_pairs import build_pairs


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class AlgebraTests(unittest.TestCase):
    def test_shared_component_retained(self):
        rng = np.random.default_rng(17)
        for alpha in (0.5, 1.0, 1.5):
            b, ec, es = rng.normal(size=(3, 17))
            actual = (b + ec) + alpha * ((b + ec) - (b + es))
            expected = b + (1 + alpha) * ec - alpha * es
            np.testing.assert_allclose(actual, expected)

    def test_shared_shift_preserves_distribution(self):
        clean = np.array([2.0, -0.5, 0.0, 0.6])
        shuffled = clean + 7.3
        for alpha in (0.5, 1.0, 1.5):
            for temperature in (0.05, 1.0, 2.0):
                corrected = clean + alpha * (clean - shuffled)
                np.testing.assert_allclose(softmax(corrected / temperature), softmax(clean / temperature))

    def test_pairwise_margin_condition(self):
        # Candidate 0 is wrong and wins on clean logits; candidate 1 is the reference.
        clean = np.array([2.0, 1.0])
        shuffled = np.array([3.0, 0.0])
        margin = clean[1] - clean[0]
        difference = margin - (shuffled[1] - shuffled[0])
        for alpha in (0.4, 0.5, 0.6):
            corrected = clean + alpha * (clean - shuffled)
            self.assertAlmostEqual(corrected[1] - corrected[0], margin + alpha * difference)
        self.assertGreater((clean + 0.6 * (clean - shuffled))[1], (clean + 0.6 * (clean - shuffled))[0])

    def test_deterministic_prompt_only_pair(self):
        gold = ['Yes.', 'No.']
        for answer in ('Yes.', 'No.'):
            correct = [answer == label for label in gold]
            self.assertEqual(sum(correct), 1)
            self.assertFalse(all(correct))
        for p in np.linspace(0, 1, 101):
            self.assertLessEqual(p * (1 - p), 0.25 + 1e-12)

    def test_explicit_parser(self):
        self.assertEqual(parse_answer('car', 'mcq'), INVALID)
        self.assertEqual(parse_answer('car', 'mcq', {'B': 'car', 'A': 'truck'}), 'B')
        self.assertEqual(parse_answer('A. Explanation', 'mcq'), 'A')
        self.assertEqual(parse_answer('No.', 'binary'), 'No.')
        self.assertEqual(parse_answer('Yes.', 'binary', error='failed inference'), INVALID)


class PairingTests(unittest.TestCase):
    def test_maximum_disjoint_matching_and_exact_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant = root / 'b4dl_dataset/hallucination_v2'
            variant.mkdir(parents=True)
            rows = []
            for scene, gold in [('s1', 'Yes.'), ('s2', 'Yes.'), ('s2', 'No.'), ('s3', 'No.')]:
                rows.append(dict(scene_id=scene, answer=gold, question_type='binary', question='q', prompt='exact q'))
            rows.extend([
                dict(scene_id='s4', answer='Yes.', question_type='binary', question='q', prompt='exact q'),
                dict(scene_id='s5', answer='No.', question_type='binary', question='q', prompt='different prompt'),
            ])
            for cat in CATS:
                (variant / f'{cat}.json').write_text(json.dumps(rows if cat == 'motion_action' else []))
            lengths = dict(s1=40, s2=40, s3=40, s4=39, s5=40)
            first = build_pairs(root, False, lengths)
            self.assertEqual(first, build_pairs(root, False, lengths))
            self.assertEqual(len(first), 2)
            used = set()
            for pair in first:
                self.assertNotEqual(pair['positive']['scene_id'], pair['negative']['scene_id'])
                for side in ('positive', 'negative'):
                    index = pair[side]['index']
                    self.assertNotIn(index, used)
                    used.add(index)
                    self.assertEqual(rows[index]['prompt'], pair['prompt'])
                    self.assertEqual(lengths[rows[index]['scene_id']], pair['sequence_length'])


class ArchivedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(os.environ.get('LIDAR_HALLU_DATA', 'data'))
        if not (cls.data / 'b4dl_dataset/hallucination_v2').exists():
            raise unittest.SkipTest('Set LIDAR_HALLU_DATA to run archive integration tests.')

    def test_pair_counts_are_prediction_independent(self):
        lengths = {}
        for cat in CATS:
            path = self.data / 'b4dl_eval/hallucination_v2_temporal_shuffle_cd' / f'{cat}_predictions.jsonl'
            for line in path.read_text().splitlines():
                row = json.loads(line)
                perm = row['shuffle_permutation']
                lengths[row['scene_id']] = len(perm.split()) if isinstance(perm, str) else len(perm)
        for meta, expected in ((False, 348), (True, 38)):
            pairs = build_pairs(self.data, meta, lengths)
            self.assertEqual(len(pairs), expected)
            self.assertEqual(len({(p['category'], p[s]['index']) for p in pairs for s in ('positive', 'negative')}), 2 * expected)

    def test_lateral_collapse_all_ten_runs(self):
        questions = json.loads((self.data / 'b4dl_dataset/hallucination_v2/motion_action.json').read_text())
        indices = [i for i, q in enumerate(questions) if q['action'] in ('moves left relative to the ego vehicle', 'moves right relative to the ego vehicle')]
        self.assertEqual(len(indices), 722)
        self.assertEqual(sum(questions[i]['answer'] == 'Yes.' for i in indices), 302)
        count = 0
        for folder in (self.data / 'b4dl_eval').iterdir():
            path = folder / 'motion_action_predictions.jsonl'
            if not path.is_file():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for i in indices:
                self.assertEqual(parse_answer(rows[i]['prediction'], 'binary', error=rows[i].get('error', '')), 'No.')
                count += 1
        self.assertEqual(count, 7220)


if __name__ == '__main__':
    unittest.main(verbosity=2)
