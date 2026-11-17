import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from peptide_lab.config import ROOT
from peptide_lab.datasets import catalog,browse,attach_labels
from peptide_lab.config import RESULTS, MODELS
import json
import hashlib

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=catalog()

    def test_filters_and_unknown_labels(self):
        frame=browse(self.catalog,'Toxicity benchmark','test','1',5,40)
        self.assertGreater(len(frame),0)
        self.assertTrue(frame.toxicity_label.eq(1).all())
        self.assertTrue(frame.toxicity_split.eq('test').all())
        self.assertTrue(frame.length.between(5,40).all())
        self.assertTrue(self.catalog.toxicity_label.isna().any())
        self.assertFalse(self.catalog.sequence.duplicated().any())

    def test_labels_join_by_sequence_not_rank(self):
        selected=pd.DataFrame({'sequence':['ACDEF','GHIKL'],'activity_label':pd.Series([1,0],dtype='Int64'),'toxicity_label':pd.Series([pd.NA,1],dtype='Int64')})
        predictions=pd.DataFrame({'sequence':['GHIKL','ACDEF'],'activity_score':[.1,.9],'toxicity_score':[.8,.2]})
        out=attach_labels(predictions,selected).set_index('sequence')
        self.assertEqual(out.loc['GHIKL','activity_label'],0)
        self.assertTrue(out.activity_matches_label.all())
        self.assertTrue(pd.isna(out.loc['ACDEF','toxicity_matches_label']))

    def test_dataset_prediction_and_history_interface(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
        app.sidebar.radio[0].set_value('Dataset explorer').run()
        self.assertFalse(app.exception)
        app.selectbox(key='dataset_source').set_value('Toxicity benchmark').run()
        app.selectbox(key='dataset_split').set_value('test').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        frame,settings=app.session_state['dataset_result']
        self.assertEqual(len(frame),3)
        self.assertTrue(frame.toxicity_split.eq('test').all())
        self.assertTrue(frame.toxicity_label.notna().all())
        app.sidebar.radio[0].set_value('Prediction history').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['prediction_runs']),1)
        app.sidebar.radio[0].set_value('Dataset explorer').run()
        app.text_input(key='dataset_search').set_value('ZZZZZZ').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

    def test_research_page_and_preserved_model_weights(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
        app.sidebar.radio[0].set_value('Research checks').run()
        self.assertFalse(app.exception)
        recorded=json.loads((RESULTS/'run_manifest.json').read_text())['model_sha256']
        for name,digest in recorded.items():
            self.assertEqual(hashlib.sha256((MODELS/name).read_bytes()).hexdigest(),digest)

    def test_controlled_study_has_matched_disjoint_splits(self):
        runs=pd.read_csv(RESULTS/'extended/toy_study_runs.csv')
        self.assertEqual(len(runs),22)
        self.assertEqual(len(runs[(runs.latent==24)&(runs.word_dropout==.2)]),10)
        folder=RESULTS/'extended/toy_runs'
        for seed in [42,101,123]:
            reference={split:pd.read_csv(folder/f'seed_{seed}_latent_24_dropout_0.2'/f'{split}.csv').sequence.tolist() for split in ['train','validation','test']}
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                self.assertFalse(set(reference[a])&set(reference[b]))
            for latent in [16,32,64]:
                for split in reference:
                    self.assertEqual(reference[split],pd.read_csv(folder/f'seed_{seed}_latent_{latent}_dropout_0.2'/f'{split}.csv').sequence.tolist())

if __name__=='__main__':
    unittest.main()
