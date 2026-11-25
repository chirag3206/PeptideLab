"""Reproducible full-split hyperparameter search; production weights stay intact.

Run: python -m peptide_lab.tuning
Trials use training/validation only. Freeze selection before loading test data.
Resume requires identical protocol, source code and split checksums.
"""
import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .config import ROOT, DATA, RESULTS, MODELS
from .features import features, encode
from .networks import PeptideCNN, PeptideVAE
from .training import seed_everything, cnn_predict, metrics, vae_arrays

OUTPUT = RESULTS/'tuning'
SPACES = {
    'cnn': {'lr':[.0005,.001], 'filters':[16,32], 'dropout':[.2,.3,.5]},
    'forest': {'n_estimators':[120,240], 'max_depth':[10,20,None], 'min_samples_leaf':[1,2,4]},
    'vae': {'latent':[16,24,32], 'lr':[.001,.002], 'word_dropout':[0.,.2]},
}
BASELINES = {
    'cnn': {'lr':.001,'filters':32,'dropout':.3},
    'forest': {'n_estimators':240,'max_depth':None,'min_samples_leaf':2},
    'vae': {'latent':24,'lr':.002,'word_dropout':.2},
}

def write_json(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)

def plan(family, count=4, seed=42):
    keys=list(SPACES[family])
    candidates=[dict(zip(keys,values)) for values in itertools.product(*(SPACES[family][k] for k in keys))]
    candidates=[c for c in candidates if c!=BASELINES[family]]
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(candidates))
    return [BASELINES[family].copy()]+[candidates[i] for i in order[:count-1]]

def choose(rows, family):
    key='validation_objective' if family=='vae' else 'validation_roc_auc'
    return min(rows,key=lambda r:(r[key] if family=='vae' else -r[key],r['trial']))

def summarize(folder):
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('*/*/trial_*/result.json'))]
    if rows:
        temp=folder/'trials.csv.tmp'
        pd.json_normalize(rows,sep='_').to_csv(temp,index=False)
        temp.replace(folder/'trials.csv')

@torch.inference_mode()
def vae_measure(model, arrays):
    model.eval();loss,total,correct,kl_sum,rows=0.,0,0,0.,0
    for x,y in DataLoader(TensorDataset(*arrays),batch_size=128):
        logits,mu,lv=model(x);mask=y.ne(0)
        loss+=nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=0,reduction='sum').item()
        total+=int(mask.sum());correct+=int(((logits.argmax(-1)==y)&mask).sum())
        kl_sum+=float(-.5*(1+lv-mu.square()-lv.exp()).sum())
        rows+=len(x)
    return {'reconstruction':loss/total,'kl':kl_sum/rows,'token_accuracy':correct/total,
            'objective':loss/total+.02*kl_sum/rows}

def train_trial(task,family,config,trial,frames,folder,args):
    target=folder/task/family/f'trial_{trial:02d}';target.mkdir(parents=True,exist_ok=True)
    if (target/'result.json').exists():
        return json.loads((target/'result.json').read_text())
    seed_everything(args.seed);start=time.perf_counter()
    write_json(target/'config.json',config)
    history=[];best_epoch=None
    if family=='vae':
        arrays={k:vae_arrays(v.sequence.tolist()) for k,v in frames.items()}
        model=PeptideVAE(latent=config['latent'],word_dropout=config['word_dropout'])
        opt=torch.optim.Adam(model.parameters(),lr=config['lr'])
        loader=DataLoader(TensorDataset(*arrays['train']),batch_size=128,shuffle=True)
        best_value=float('inf');best=None;stale=0
        for epoch in range(1,args.vae_epochs+1):
            model.train();loss_total,tokens=0.,0
            beta=.02*min(1.,epoch/10)
            for x,y in loader:
                logits,mu,lv=model(x);n=int(y.ne(0).sum())
                recon=nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=0,reduction='sum')
                kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
                opt.zero_grad();(recon/n+beta*kl).backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                loss_total+=recon.item();tokens+=n
            val=vae_measure(model,arrays['validation'])
            history.append({'epoch':epoch,'beta':beta,'train_reconstruction':loss_total/tokens,**{'validation_'+k:v for k,v in val.items()}})
            if val['objective']<best_value:
                best_value=val['objective'];best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            pd.DataFrame(history).to_csv(target/'history.csv',index=False)
            if epoch%5==0:print(task,family,trial,'epoch',epoch,'validation objective',round(val['objective'],5),flush=True)
            if stale>=7:break
        model.load_state_dict(best);torch.save(best,target/'weights.pt')
        validation={'validation_'+k:v for k,v in vae_measure(model,arrays['validation']).items()}
    else:
        xs={k:features(v.sequence.tolist()) for k,v in frames.items()}
        ys={k:v.label.to_numpy(dtype=np.float32) for k,v in frames.items()}
        if family=='forest':
            model=RandomForestClassifier(**config,max_features=.7,class_weight='balanced',n_jobs=4,random_state=args.seed)
            model.fit(xs['train'],ys['train']);pred=model.predict_proba(xs['validation'])[:,1]
            joblib.dump(model,target/'forest.joblib')
        else:
            scaler=StandardScaler().fit(xs['train'])
            scaled={k:scaler.transform(v).astype(np.float32) for k,v in xs.items()}
            tokens={k:encode(v.sequence.tolist()) for k,v in frames.items()}
            model=PeptideCNN(filters=config['filters'],dropout=config['dropout'])
            opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=.001)
            loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((ys['train']==0).sum()/(ys['train']==1).sum())))
            loader=DataLoader(TensorDataset(torch.as_tensor(tokens['train']),torch.as_tensor(scaled['train']),torch.as_tensor(ys['train'])),batch_size=128,shuffle=True)
            best_value=-1.;best=None;stale=0
            for epoch in range(1,args.cnn_epochs+1):
                model.train();loss_total=0.
                for x,d,y in loader:
                    opt.zero_grad();loss=loss_fn(model(x,d),y);loss.backward();opt.step()
                    loss_total+=loss.item()*len(y)
                pred=cnn_predict(model,tokens['validation'],scaled['validation'])
                auc=metrics(ys['validation'],pred)['roc_auc']
                history.append({'epoch':epoch,'training_loss':loss_total/len(ys['train']),'validation_auc':auc})
                if auc>best_value+.0001:
                    best_value=auc;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                else:stale+=1
                pd.DataFrame(history).to_csv(target/'history.csv',index=False)
                if epoch%5==0:print(task,family,trial,'epoch',epoch,'validation AUC',round(auc,5),flush=True)
                if stale>=5:break
            model.load_state_dict(best)
            pred=cnn_predict(model,tokens['validation'],scaled['validation'])
            torch.save(best,target/'weights.pt');joblib.dump(scaler,target/'scaler.joblib')
        validation={'validation_'+k:v for k,v in metrics(ys['validation'],pred).items() if k!='confusion_matrix'}
        saved=frames['validation'].copy();saved['score']=pred;saved.to_csv(target/'validation_predictions.csv',index=False)
    row={'task':task,'family':family,'trial':trial,'role':'baseline' if trial==0 else 'candidate',
         'config':config,'best_epoch':best_epoch,'seconds':time.perf_counter()-start,'seed':args.seed,**validation}
    write_json(target/'result.json',row);summarize(folder)
    print('Completed',task,family,trial,validation,flush=True)
    return row

def predict_trial(target,family,config,frame):
    if family=='forest':return joblib.load(target/'forest.joblib').predict_proba(features(frame.sequence.tolist()))[:,1]
    model=PeptideCNN(filters=config['filters'],dropout=config['dropout'])
    model.load_state_dict(torch.load(target/'weights.pt',weights_only=True,map_location='cpu'))
    scaler=joblib.load(target/'scaler.joblib')
    return cnn_predict(model,encode(frame.sequence.tolist()),scaler.transform(features(frame.sequence.tolist())).astype(np.float32))

def evaluate_selected(folder,selections,tasks):
    comparisons=[]
    for task in tasks:
        # This function runs only AFTER selection.json is committed to disk.
        test=pd.read_csv(DATA/'final'/f'{task}_test.csv')
        if task=='vae':
            results={}
            for stage,trial in [('Before',0),('After',selections['vae']['vae']['trial'])]:
                target=folder/'vae/vae'/f'trial_{trial:02d}';config=json.loads((target/'config.json').read_text())
                model=PeptideVAE(latent=config['latent'],word_dropout=config['word_dropout'])
                model.load_state_dict(torch.load(target/'weights.pt',weights_only=True,map_location='cpu'))
                value=vae_measure(model,vae_arrays(test.sequence.tolist()))
                seqs,ended=model.sample(500,seed=42,temperature=.9)
                known=set(pd.read_csv(DATA/'processed/grampa_clean_unique_sequences.csv').sequence);unique=set(seqs)
                value.update(uniqueness=len(unique)/len(seqs),exact_novelty=sum(s not in known for s in unique)/len(unique),natural_end_rate=float(np.mean(ended)))
                pd.DataFrame({'sequence':seqs,'ended_naturally':ended}).to_csv(folder/f'vae_{stage.lower()}_generation.csv',index=False)
                results[stage]=value
            for metric in results['Before']:
                comparisons.append({'task':task,'component':'vae','metric':metric,'before':results['Before'][metric],'after':results['After'][metric],'delta':results['After'][metric]-results['Before'][metric]})
            write_json(folder/'vae_test_results.json',results)
            continue
        predictions={};test_results={}
        for stage in ['Before','After']:
            predictions[stage]={}
            for family in ['cnn','forest']:
                trial=0 if stage=='Before' else selections[task][family]['trial']
                target=folder/task/family/f'trial_{trial:02d}';config=json.loads((target/'config.json').read_text())
                predictions[stage][family]=predict_trial(target,family,config,test)
            predictions[stage]['ensemble']=(predictions[stage]['cnn']+predictions[stage]['forest'])/2
            selected_name=selections[task]['pipeline'][stage.lower()]['model']
            predictions[stage]['selected_pipeline']=predictions[stage][selected_name]
            test_results[stage]={name:metrics(test.label,pred) for name,pred in predictions[stage].items()}
        exported=test.copy()
        for stage in predictions:
            for name,pred in predictions[stage].items():exported[stage.lower()+'_'+name+'_score']=pred
        exported.to_csv(folder/f'{task}_test_predictions.csv',index=False)
        write_json(folder/f'{task}_test_results.json',test_results)
        for component in ['cnn','forest','selected_pipeline']:
            for metric in ['accuracy','precision','recall','f1','roc_auc','mcc']:
                before=test_results['Before'][component][metric];after=test_results['After'][component][metric]
                comparisons.append({'task':task,'component':component,'metric':metric,'before':before,'after':after,'delta':after-before})
    pd.DataFrame(comparisons).to_csv(folder/'before_after.csv',index=False)

def run(args):
    if not 2<=args.trials<=12:raise ValueError('Use 2–12 trials, including baseline.')
    if min(args.cnn_epochs,args.vae_epochs)<1:raise ValueError('Epoch budgets must be positive.')
    folder=Path(args.output).resolve();folder.mkdir(parents=True,exist_ok=True)
    tasks=['activity','toxicity','vae'] if args.component=='all' else [args.component]
    protocol={'version':1,'seed':args.seed,'trials_per_family':args.trials,'cnn_epochs':args.cnn_epochs,'vae_epochs':args.vae_epochs,
              'tasks':tasks,'search':'Seeded random sample without replacement, baseline always included; full saved training/validation splits.',
              'selection':'Validation ROC-AUC for classifiers; reconstruction + fixed 0.02*KL for VAE. Ties retain earliest trial.',
              'limitations':'Single training seed. Follow-up on previously inspected test benchmarks; not external or similarity-disjoint validation. ESM excluded from this search.',
              'spaces':SPACES,'plans':{f:plan(f,args.trials,args.seed) for f in SPACES},
              'split_sha256':{f'{task}_{part}':hashlib.sha256((DATA/'final'/f'{task}_{part}.csv').read_bytes()).hexdigest() for task in tasks for part in ['train','validation','test']},
              'code_sha256':{name:hashlib.sha256((ROOT/'peptide_lab'/name).read_bytes()).hexdigest() for name in ['tuning.py','networks.py','features.py','training.py']},
              'production_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.is_file()}}
    manifest=folder/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Protocol differs from existing run. Use a different --output directory.')
    write_json(manifest,protocol)
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('state')=='complete':
        print('Matching search already complete:',folder);return
    started=time.time();selections={}
    try:
        for task in tasks:
            frames={part:pd.read_csv(DATA/'final'/f'{task}_{part}.csv') for part in ['train','validation']}
            selections[task]={}
            for family in (['vae'] if task=='vae' else ['cnn','forest']):
                rows=[]
                for trial,config in enumerate(protocol['plans'][family]):
                    write_json(folder/'status.json',{'state':'running','task':task,'family':family,'trial':trial,'trials_per_family':args.trials})
                    rows.append(train_trial(task,family,config,trial,frames,folder,args))
                winner=choose(rows,family);selections[task][family]=winner
            if task!='vae':
                selections[task]['pipeline']={}
                for stage in ['before','after']:
                    scores={}
                    for family in ['cnn','forest']:
                        trial=0 if stage=='before' else selections[task][family]['trial']
                        path=folder/task/family/f'trial_{trial:02d}'/'validation_predictions.csv'
                        scores[family]=pd.read_csv(path).score.to_numpy()
                    scores['ensemble']=(scores['cnn']+scores['forest'])/2
                    aucs={name:metrics(frames['validation'].label,p)['roc_auc'] for name,p in scores.items()}
                    selections[task]['pipeline'][stage]={'model':max(aucs,key=aucs.get),'validation_auc':aucs}
        write_json(folder/'selection.json',selections)
        write_json(folder/'status.json',{'state':'evaluating','note':'Selection frozen; evaluating baseline and selected configurations on test splits.'})
        evaluate_selected(folder,selections,tasks)
        for name,expected in protocol['production_sha256'].items():
            if hashlib.sha256((MODELS/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Production model changed during tuning: '+name)
        write_json(folder/'status.json',{'state':'complete','seconds_this_invocation':time.time()-started,'production_models_unchanged':True})
        print('Tuning complete:',folder,flush=True)
    except Exception as exc:
        write_json(folder/'status.json',{'state':'failed','error':str(exc)});raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=['all','activity','toxicity','vae'],default='all')
    parser.add_argument('--trials',type=int,default=4)
    parser.add_argument('--cnn-epochs',type=int,default=18)
    parser.add_argument('--vae-epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default=str(OUTPUT))
    run(parser.parse_args())

if __name__=='__main__':main()
