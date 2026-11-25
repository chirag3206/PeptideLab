import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from peptide_lab.config import ROOT
from peptide_lab.pipeline import ScreeningPipeline, priority_scores
from peptide_lab.targets import BUNDLE, rank_target


class PredictionModeTests(unittest.TestCase):
    def assert_ranking(self, rows, rule, weight, target=False):
        ranked = rows.sort_values('general_rank' if target else 'rank')
        activity = ranked.general_activity_score if target else ranked.activity_score
        toxicity = ranked.toxicity_score
        def expected(a, t):
            return 100 * (a * (1 - t) if rule == 'product' else weight*a + (1-weight)*(1-t))
        np.testing.assert_allclose(ranked.general_priority_score if target else ranked.priority_score,
                                   expected(activity, toxicity), atol=1e-5)
        self.assertTrue((ranked.general_priority_score if target else ranked.priority_score).is_monotonic_decreasing)
        self.assertEqual(rows['rank'].tolist(), list(range(1, len(rows) + 1)))
        if target:
            np.testing.assert_allclose(rows.priority_score, expected(rows.target_activity_score, rows.toxicity_score), atol=1e-5)
            self.assertTrue(rows.priority_score.is_monotonic_decreasing)

    def test_live_target_heads_match_held_out_scores_for_every_organism(self):
        targets = joblib.load(BUNDLE)['targets']
        for engine in ('compact', 'esm2'):
            pipe = ScreeningPipeline(engine=engine)
            score_column = 'target_esm2_score' if engine == 'esm2' else 'target_activity_score'
            for code, entry in targets.items():
                if 'test' not in entry:
                    continue
                with self.subTest(engine=engine, organism=code):
                    held = entry['test'].head(3).set_index('sequence')
                    predicted = pipe.predict(held.index.tolist(), target_entry=entry)
                    result = rank_target(predicted, code, entry, pipe.ranking_rule, pipe.activity_weight)
                    self.assertTrue(result.target_engine.eq(engine).all())
                    np.testing.assert_allclose(result.set_index('sequence').loc[held.index, 'target_activity_score'],
                                               held[score_column], atol=1e-5)

    def test_sidebar_engines_and_rules_across_input_routes(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
        for engine, option in [('compact', 'Compact baseline (fast)'), ('esm2', 'ESM-2 + CNN')]:
            app.selectbox(key='prediction_engine_choice').set_value(option).run()
            previous = {}
            for rule, label in [('product', 'Product'), ('weighted_difference', 'Weighted difference')]:
                app.selectbox(key='rank_rule').set_value(label).run()
                if rule == 'weighted_difference':
                    app.slider(key='rank_weight').set_value(.75).run()
                for page, modes in [('Dataset explorer', ['GRAMPA', 'Activity benchmark', 'Toxicity benchmark']),
                                    ('Screen sequences', ['custom']), ('Generate & rank', ['generated']),
                                    ('Target-specific screening', ['Dataset peptides','Enter sequences','Generate peptides'])]:
                    app.sidebar.radio[0].set_value(page).run()
                    for mode in modes:
                        with self.subTest(engine=engine, rule=rule, page=page, mode=mode):
                            target = page == 'Target-specific screening'
                            if target:
                                app.radio(key='target_mode').set_value(mode).run()
                                if mode == 'Enter sequences':
                                    app.text_area(key='target_input').set_value('ACDEFGHIK\nLMNPQRSTV')
                            elif page == 'Dataset explorer':
                                app.selectbox(key='dataset_source').set_value(mode).run()
                            app.button[0].click().run()
                            self.assertFalse(app.exception)
                            rows = app.session_state['prediction_runs'][-1]['frame']
                            self.assertGreater(len(rows), 0)
                            self.assertTrue(rows.ranking_rule.eq(rule).all())
                            self.assertTrue((rows.target_engine if target else rows.prediction_engine).eq(engine).all())
                            self.assert_ranking(rows, rule, .75, target)
                            fields = (['general_activity_score', 'target_activity_score'] if target else ['activity_score']) + ['toxicity_score']
                            scores = rows.set_index('sequence')[fields].sort_index()
                            if rule == 'product':
                                previous[page, mode] = scores
                            else:
                                pd.testing.assert_frame_equal(scores, previous[page, mode], atol=1e-5, rtol=1e-5)

    def test_weight_slider_endpoints(self):
        for weight, expected in [(0, 80), (1, 60)]:
            self.assertAlmostEqual(priority_scores(.6, .2, 'weighted_difference', weight), expected)
