import json
import unittest
import numpy as np
import pandas as pd
from peptide_lab.config import ROOT, DATA, RESULTS
from peptide_lab.pipeline import ScreeningPipeline

class FinalAlignmentTests(unittest.TestCase):
    def test_weighted_ranking_and_property_flags(self):
        pipe = ScreeningPipeline(ranking_rule='weighted_difference', activity_weight=.7)
        frame, audit = pipe.generate(100, seed=24)
        np.testing.assert_allclose(frame.priority_score,100*(.7*frame.activity_score+.3*(1-frame.toxicity_score)))
        self.assertTrue(frame.priority_score.is_monotonic_decreasing)
        self.assertEqual(frame['rank'].tolist(), list(range(1,len(frame)+1)))
        expected=(frame.activity_score>=frame.activity_screen_threshold)&(frame.toxicity_score<frame.toxicity_screen_threshold)
        self.assertEqual(expected.tolist(),frame.passes_property_screen.tolist())
        self.assertEqual(int(expected.sum()),audit['passed_property_screen'])
        self.assertEqual(len(frame),audit['passed_filters'])

    def test_esm_saved_metrics_and_live_predictions(self):
        from peptide_lab.training import metrics
        from peptide_lab.esm_predictor import ESMPredictor
        predictor=ESMPredictor()
        for task in ['activity','toxicity']:
            frame=pd.read_csv(RESULTS/f'esm2/{task}_test_predictions.csv')
            report=json.loads((RESULTS/f'esm2/{task}_metrics.json').read_text())
            observed=metrics(frame.label,frame.esm2_score)
            self.assertAlmostEqual(observed['roc_auc'],report['test']['roc_auc'])
            self.assertEqual(observed['confusion_matrix'],report['test']['confusion_matrix'])
            np.testing.assert_allclose(predictor.predict(frame.sequence.head(3).tolist())[task],frame.esm2_score.head(3),atol=1e-5)
        sequences=frame.sequence.head(2).tolist()
        fresh=predictor.embedder.encode(sequences)
        cached=np.array([predictor.cached[predictor.lookup[s]] for s in sequences])
        np.testing.assert_allclose(fresh,cached,atol=1e-5,rtol=1e-4)

    def test_validation_threshold_provenance(self):
        import hashlib
        reports=json.loads((RESULTS/'alignment/property_thresholds.json').read_text())
        for engine, tasks in reports.items():
            for task, report in tasks.items():
                path=DATA/f'final/{task}_validation.csv'
                self.assertEqual(report['validation_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                if task=='toxicity':self.assertGreaterEqual(report['recall'],.95)

    def test_new_dashboard_pages_and_conditional_controls(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        for page in ['Model results','Dataset explorer']:
            at.sidebar.radio[0].set_value(page).run()
            self.assertFalse(at.exception, str(at.exception))
        at.selectbox(key='dataset_batch_mode').set_value('Random sample of filtered dataset').run()
        self.assertFalse(any(w.label=='Select peptides from this page' for w in at.multiselect))
        self.assertTrue(any(w.label=='Random sample seed' for w in at.number_input))
        at.selectbox(key='dataset_batch_mode').set_value('First rows of filtered dataset').run()
        self.assertFalse(any(w.label=='Random sample seed' for w in at.number_input))
        self.assertFalse(at.exception, str(at.exception))
