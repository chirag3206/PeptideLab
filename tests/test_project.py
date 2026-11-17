"""Behavioral and integrity checks for the actual demonstration pipeline."""
import json
import unittest
import numpy as np
import pandas as pd
from peptide_lab.config import DATA, MODELS, RESULTS
from peptide_lab.features import validate, parse_sequences, properties

class InputTests(unittest.TestCase):
    def test_fasta_and_duplicates(self):
        self.assertEqual(parse_sequences('>one\nACDEF\nGHIK\n>two\nACDEFGHIK'), ['ACDEFGHIK'])

    def test_invalid_input_rejected(self):
        for sequence in ['ABCDE', '', 'ACD', 'A' * 201, 'ACD*FG']:
            with self.assertRaises(ValueError):
                validate(sequence)

    def test_normalization_and_descriptors(self):
        self.assertEqual(validate(' acdef ghik '), 'ACDEFGHIK')
        self.assertEqual(properties('KKRDD')['charge_proxy'], 1)

class DataIntegrityTests(unittest.TestCase):
    def test_splits_do_not_share_exact_sequences(self):
        for task in ['activity', 'toxicity', 'vae']:
            sets = [set(pd.read_csv(DATA / 'final' / f'{task}_{split}.csv').sequence) for split in ['train', 'validation', 'test']]
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                self.assertFalse(sets[a] & sets[b], task)

    def test_metrics_recomputed_from_predictions(self):
        from peptide_lab.training import metrics
        for task in ['activity', 'toxicity']:
            report = json.loads((RESULTS / f'{task}_metrics.json').read_text())
            frame = pd.read_csv(RESULTS / f'{task}_test_predictions.csv')
            selected = report['selected_model']
            self.assertEqual(selected, max(report['validation_auc'], key=report['validation_auc'].get))
            observed = metrics(frame.label, frame[selected + '_score'])
            self.assertAlmostEqual(observed['roc_auc'], report['test'][selected]['roc_auc'])
            self.assertEqual(observed['confusion_matrix'], report['test'][selected]['confusion_matrix'])

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peptide_lab.pipeline import ScreeningPipeline
        cls.pipeline = ScreeningPipeline()

    def test_predictions_and_ranking(self):
        frame = self.pipeline.predict(['ACDEFGHIK', 'GILSGILGAGKSLVCGLSGLC'])
        self.assertEqual(len(frame), 2)
        for column in ['activity_score', 'toxicity_score']:
            self.assertTrue(frame[column].between(0, 1).all())
        np.testing.assert_allclose(frame.priority_score, 100 * frame.activity_score * (1-frame.toxicity_score))
        self.assertTrue(frame.priority_score.is_monotonic_decreasing)

    def test_generated_candidates_are_reproducible_valid_and_novel(self):
        first, audit = self.pipeline.generate(5, seed=17)
        second, _ = self.pipeline.generate(5, seed=17)
        self.assertGreater(len(first), 0)
        self.assertEqual(first.sequence.tolist(), second.sequence.tolist())
        self.assertEqual(len(first), first.sequence.nunique())
        self.assertFalse(first.seen_in_grampa.any())
        self.assertFalse(first.seen_in_classifier_training.any())
        self.assertTrue(first.length.between(10, 40).all())
        self.assertEqual(audit['sampled'], audit['accepted_before_limit'] + sum(audit['rejected'].values()))

if __name__ == '__main__':
    unittest.main()
