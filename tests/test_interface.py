import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
    def test_overall_and_organism_demo(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
        next(item for item in app.button if item.label == 'Open target-specific screening').click().run()
        self.assertFalse(app.exception)
        app.selectbox(key='target_organism').set_value('E. coli').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        rows = app.session_state['target_last_result'][2]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted(rows.general_rank), [1, 2, 3])
        self.assertTrue(rows.observed_lower_MIC_label.isin([0, 1]).all())
        self.assertTrue(rows.general_activity_score.between(0, 1).all())
        self.assertTrue(rows.target_activity_score.between(0, 1).all())
        app.sidebar.radio[0].set_value('Prediction history').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.dataframe[0].value.equals(rows))
        app.sidebar.radio[0].set_value('Target-specific screening').run()
        app.radio(key='target_view').set_value('Overall results').run()
        self.assertFalse(app.exception)
        general, targets = [table.value for table in app.dataframe]
        self.assertEqual(set(general.Task), {'Activity', 'Toxicity'})
        self.assertEqual(targets['ROC-AUC'].count(), 8)
        self.assertTrue(targets.loc[targets.Status == 'Evidence only', 'Accuracy'].isna().all())
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value, 'Screen sequences')

    def test_navigation_screening_and_generation(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
        self.assertFalse(app.exception)
        for page in ['Model results', 'Learn the project', 'Screen sequences']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, page)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn('screened', app.session_state)
        self.assertEqual(len(app.session_state['screened']), 2)
        app.text_area[0].set_value('INVALID123')
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn('screened', app.session_state)
        app.sidebar.radio[0].set_value('Generate & rank').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        frame, audit = app.session_state['generated']
        self.assertGreater(len(frame), 0)
        self.assertEqual(len(frame), audit['passed_filters'])
        self.assertEqual(frame['rank'].tolist(), list(range(1, len(frame)+1)))

if __name__ == '__main__':
    unittest.main()
