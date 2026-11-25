"""Search selection, configurable architecture and dashboard behaviour."""
import json
import unittest
import numpy as np
import pandas as pd
import torch
from peptide_lab.config import ROOT, RESULTS, MODELS
from peptide_lab.networks import PeptideCNN
from peptide_lab.tuning import BASELINES, plan, choose

class TuningTests(unittest.TestCase):
    def test_search_includes_baseline_and_unique_repeatable_candidates(self):
        for family in BASELINES:
            configs=plan(family,4,42)
            self.assertEqual(configs[0],BASELINES[family])
            self.assertEqual(configs,plan(family,4,42))
            self.assertEqual(len(configs),4)
            self.assertEqual(len({json.dumps(c,sort_keys=True) for c in configs}),4)

    def test_selection_uses_validation_and_prefers_baseline_on_tie(self):
        rows=[{'trial':0,'validation_roc_auc':.9,'test_accuracy':.1},
              {'trial':1,'validation_roc_auc':.8,'test_accuracy':1.},
              {'trial':2,'validation_roc_auc':.9,'test_accuracy':.99}]
        self.assertEqual(choose(rows,'cnn')['trial'],0)
        self.assertEqual(choose([{'trial':0,'validation_objective':2.}, {'trial':1,'validation_objective':1.5}],'vae')['trial'],1)

    def test_configurable_cnn_and_old_weights_compatible(self):
        default=PeptideCNN()
        default.load_state_dict(torch.load(MODELS/'activity_cnn.pt',map_location='cpu',weights_only=True))
        x=torch.tensor([[1,2,3,4,5,0,0],[4,5,6,7,8,9,0]])
        for filters in [16,32]:
            model=PeptideCNN(filters=filters,dropout=.5).eval()
            out=model(x,torch.zeros(2,25))
            self.assertEqual(out.shape,(2,));self.assertTrue(torch.isfinite(out).all())

    def test_tuning_page(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
        self.assertNotIn('Hyperparameter tuning',app.sidebar.radio[0].options)
        app.sidebar.radio[0].set_value('Model results').run()
        for task in ['activity','toxicity']:
            app.radio(key='results_view_'+task).set_value('Hyperparameter tuning').run()
            self.assertFalse(app.exception,str(app.exception))
            self.assertIsNotNone(app.selectbox(key='tuning_family_'+task))
            tables=[d.value for d in app.dataframe if 'Configuration' in d.value.columns]
            self.assertTrue(any(len(frame)==4 and frame.iloc[0]['Configuration']=='Default' for frame in tables))
        # Both classifier tuning views can coexist.
        app.selectbox(key='tuning_family_activity').set_value('forest').run()
        self.assertFalse(app.exception,str(app.exception))
        tables=[d.value for d in app.dataframe if 'Configuration' in d.value.columns]
        self.assertTrue(any('Trees' in frame.columns for frame in tables))
        self.assertTrue(any('Final verdict' in m.value for m in app.markdown))
        self.assertFalse(any('Search protocol' == tab.label for tab in app.tabs))
        app.selectbox(key='tuning_family_activity').set_value('ensemble').run()
        self.assertFalse(app.exception,str(app.exception))
        tables=[d.value for d in app.dataframe if 'CNN weight' in d.value.columns]
        self.assertEqual(len(tables),1)
        self.assertEqual(len(tables[0]),2)
        self.assertNotIn('ensemble',app.selectbox(key='tuning_family_toxicity').options)
        for task in ['activity','toxicity']:
            app.radio(key='results_view_'+task).set_value('Baseline results').run()
            self.assertFalse(app.exception,str(app.exception))

        app.radio(key='results_section').set_value('VAE generator').run()
        app.radio(key='results_view_vae').set_value('Hyperparameter tuning').run()
        self.assertFalse(app.exception)
        self.assertIsNotNone(app.selectbox(key='tuning_family_vae'))

    def test_readable_tables_match_saved_results(self):
        from peptide_lab.tuning_dashboard import comparison_table, test_comparison
        folder=RESULTS/'tuning'
        selections=json.loads((folder/'selection.json').read_text())
        for task, families in selections.items():
            for family, selected in families.items():
                if family=='pipeline':continue
                rows=[json.loads(p.read_text()) for p in (folder/task/family).glob('trial_*/result.json')]
                table, baseline, winner=comparison_table(rows,family)
                self.assertEqual(winner['trial'],selected['trial'])
                self.assertEqual((table.Verdict=='Best tested (selected)').sum(),1)
                self.assertEqual(table.iloc[0]['Changes from default'],'Default settings (no changes)')
                if family=='forest':self.assertEqual(table.iloc[0]['Maximum depth'],'Unlimited')
                comparison=test_comparison(folder,task,family)
                self.assertFalse(comparison.empty)
                np.testing.assert_allclose(comparison['Selected configuration']-comparison.Default,comparison.Change,atol=1e-10)

    def test_ensemble_comparison_matches_saved_predictions(self):
        from peptide_lab.tuning_dashboard import ensemble_table, test_comparison
        from peptide_lab.training import metrics
        folder=RESULTS/'tuning'
        table,selection=ensemble_table(folder,'activity')
        for stage,key in [(0,'before'),(1,'after')]:
            self.assertAlmostEqual(table.iloc[stage]['Validation ROC-AUC ↑'],selection['pipeline'][key]['validation_auc']['ensemble'])
        comparison=test_comparison(folder,'activity','ensemble')
        predictions=pd.read_csv(folder/'activity_test_predictions.csv')
        for stage in ['before','after']:
            np.testing.assert_allclose(predictions[stage+'_ensemble_score'],(predictions[stage+'_cnn_score']+predictions[stage+'_forest_score'])/2,atol=1e-12)
            observed=metrics(predictions.label,predictions[stage+'_ensemble_score'])
            for metric,label,scale in [('accuracy','Accuracy (%) ↑',100),('roc_auc','ROC-AUC ↑',1),('f1','F1 ↑',1)]:
                row=comparison[comparison.Metric==label].iloc[0]
                column='Default' if stage=='before' else 'Selected configuration'
                self.assertAlmostEqual(row[column],scale*observed[metric])

    def test_completed_search_evidence_if_available(self):
        folder=RESULTS/'tuning'
        if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['state']!='complete':
            self.skipTest('Full tuning search has not finished yet.')
        from peptide_lab.training import metrics
        selections=json.loads((folder/'selection.json').read_text())
        comparison=pd.read_csv(folder/'before_after.csv')
        import hashlib
        protocol=json.loads((folder/'protocol.json').read_text())
        for name,expected in protocol['production_sha256'].items():
            self.assertEqual(hashlib.sha256((MODELS/name).read_bytes()).hexdigest(),expected)
        for task,families in selections.items():
            for family,winner in families.items():
                if family=='pipeline':continue
                rows=[json.loads(p.read_text()) for p in (folder/task/family).glob('trial_*/result.json')]
                self.assertEqual(choose(rows,family)['trial'],winner['trial'])
            if task=='vae':
                measured=json.loads((folder/'vae_test_results.json').read_text())
                for stage in ['Before','After']:
                    for metric,value in measured[stage].items():
                        row=comparison[(comparison.task=='vae')&(comparison.metric==metric)].iloc[0]
                        self.assertAlmostEqual(value,row[stage.lower()])
                continue
            frame=pd.read_csv(folder/f'{task}_test_predictions.csv')
            for stage in ['before','after']:
                measured=metrics(frame.label,frame[stage+'_selected_pipeline_score'])
                for metric in ['accuracy','f1','roc_auc']:
                    row=comparison[(comparison.task==task)&(comparison.component=='selected_pipeline')&(comparison.metric==metric)].iloc[0]
                    self.assertAlmostEqual(measured[metric],row[stage])
        np.testing.assert_allclose(comparison['after']-comparison['before'],comparison.delta,atol=1e-12)
