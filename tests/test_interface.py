import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class InterfaceTests(unittest.TestCase):
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
