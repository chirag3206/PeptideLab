"""Additional fixed-model diagnostics. Does not tune or replace deployed models."""
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, balanced_accuracy_score
from .config import DATA, MODELS, RESULTS, END, START, AA
from .preprocessing import prepare
from .networks import PeptideVAE
from .training import vae_arrays

def decode(ids):
    letters = []
    for token in ids:
        if token == END:
            break
        if 1 <= token <= 20:
            letters.append(AA[token-1])
    return ''.join(letters)

@torch.no_grad()
def latent_diagnostics():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = PeptideVAE()
    model.load_state_dict(torch.load(MODELS/'vae.pt',weights_only=True,map_location='cpu'))
    model.eval()
    frame = pd.read_csv(DATA/'final/vae_validation.csv')
    x,y = vae_arrays(frame.sequence.tolist())
    mu,logvar = model.encode(x)
    permutation = torch.randperm(len(mu),generator=torch.Generator().manual_seed(42))
    rows = []
    for name,z in [('encoder_mean',mu),('zero_latent',torch.zeros_like(mu)),('shuffled_latent',mu[permutation])]:
        logits,_ = model.decode(x,z)
        predicted = logits.argmax(-1)
        mask = y.ne(0)
        ce = torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=0,reduction='sum')/mask.sum()
        decoded = [decode(row) for row in predicted.tolist()]
        rows.append({'latent_condition':name,'cross_entropy':float(ce),'token_accuracy':float(((predicted==y)&mask).sum()/mask.sum()),'teacher_forced_exact_match':float(np.mean([a==b for a,b in zip(decoded,frame.sequence)]))})
    # Free-running reconstruction from the input's latent mean, without true prefixes.
    token = torch.full((len(x),1),START,dtype=torch.long)
    state, generated = None, [[] for _ in range(len(x))]
    done = [False]*len(x)
    for position in range(61):
        logits,state=model.decode(token,mu,state)
        logits[:,0,[0,START]]=-1e9
        token=logits[:,0].argmax(-1,keepdim=True)
        for i,value in enumerate(token[:,0].tolist()):
            if not done[i]:
                if value==END:
                    done[i]=True
                elif position<60:
                    generated[i].append(AA[value-1])
    sequences=[''.join(s) for s in generated]
    examples = pd.DataFrame({'original':frame.sequence,'free_running_reconstruction':sequences,'exact_match':[a==b for a,b in zip(sequences,frame.sequence)],'ended_naturally':done})
    examples.to_csv(RESULTS/'extended/reconstruction_examples.csv',index=False)
    rows.append({'latent_condition':'free_running_from_encoder_mean','cross_entropy':None,'token_accuracy':None,'teacher_forced_exact_match':None,'free_running_exact_match':float(examples.exact_match.mean())})
    return rows,int((mu.var(dim=0)>.01).sum())

def predictor_checks():
    checks,intervals = [],[]
    rng=np.random.default_rng(42)
    for task in ['activity','toxicity']:
        report=json.loads((RESULTS/f'{task}_metrics.json').read_text())
        selected=report['selected_model']
        frame=pd.read_csv(RESULTS/f'{task}_test_predictions.csv')
        train=pd.read_csv(DATA/f'final/{task}_train.csv')
        y=frame.label.to_numpy(); scores=frame[selected+'_score'].to_numpy()
        lengths=frame.sequence.str.len()
        baseline=LogisticRegression(random_state=42).fit(train.sequence.str.len().to_numpy().reshape(-1,1),train.label)
        lp=baseline.predict_proba(lengths.to_numpy().reshape(-1,1))[:,1]
        for label,mask,p in [('selected_all',np.ones(len(frame),dtype=bool),scores),('length_only_all',np.ones(len(frame),dtype=bool),lp),('selected_length_5_40',lengths.between(5,40).to_numpy(),scores)]:
            if len(np.unique(y[mask]))<2:
                continue
            checks.append({'task':task,'evaluation':label,'n':int(mask.sum()),'positive':int((y[mask]==1).sum()),'negative':int((y[mask]==0).sum()),'accuracy':accuracy_score(y[mask],p[mask]>=.5),'balanced_accuracy':balanced_accuracy_score(y[mask],p[mask]>=.5),'roc_auc':roc_auc_score(y[mask],p[mask]),'brier_score':brier_score_loss(y[mask],p[mask])})
        boot={'accuracy':[],'roc_auc':[]}
        groups=[np.flatnonzero(y==value) for value in [0,1]]
        for _ in range(300):
            idx=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
            boot['accuracy'].append(accuracy_score(y[idx],scores[idx]>=.5))
            boot['roc_auc'].append(roc_auc_score(y[idx],scores[idx]))
        for name,values in boot.items():
            low,high=np.quantile(values,[.025,.975])
            intervals.append({'task':task,'metric':name,'lower_95':float(low),'upper_95':float(high),'bootstrap_replicates':300})
        frame['selected_score']=scores
        frame['prediction']=(scores>=.5).astype(int)
        frame['error_type']=np.where(frame.prediction>frame.label,'false_positive','false_negative')
        frame[frame.prediction!=frame.label].to_csv(RESULTS/f'extended/{task}_errors.csv',index=False)
        bins=pd.cut(scores,np.linspace(0,1,11),include_lowest=True)
        reliability=pd.DataFrame({'bin':bins,'score':scores,'label':y}).groupby('bin',observed=True).agg(count=('label','size'),mean_score=('score','mean'),observed_positive_fraction=('label','mean')).reset_index()
        reliability.to_csv(RESULTS/f'extended/{task}_reliability.csv',index=False)
    return checks,intervals

def ranking_checks():
    frame=pd.read_csv(RESULTS/'ranked_candidates.csv')
    product=frame.set_index('sequence').priority_score.rank(ascending=False,method='first')
    top=set(product.nsmallest(10).index)
    rows=[]
    for weight in [.25,.5,.75]:
        score=weight*frame.activity_score-(1-weight)*frame.toxicity_score
        rank=pd.Series(score.to_numpy(),index=frame.sequence).rank(ascending=False,method='first')
        rows.append({'activity_weight':weight,'toxicity_weight':1-weight,'top_10_overlap_with_product':len(top&set(rank.nsmallest(10).index))/min(10,len(top)),'spearman_rank_correlation':float(product.corr(rank))})
    return rows

def main():
    prepare()
    folder=RESULTS/'extended'
    folder.mkdir(exist_ok=True)
    latent,active=latent_diagnostics()
    checks,intervals=predictor_checks()
    summary={'latent_diagnostics':latent,'active_latent_dimensions':active,'predictor_checks':checks,'confidence_intervals':intervals,'ranking_sensitivity':ranking_checks(),
             'protocol':'Post-hoc diagnostics of frozen existing models; no model selection or retraining. VAE diagnostics use validation; classifier diagnostics reuse the existing test predictions.',
             'source_model_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in MODELS.iterdir() if p.is_file()}}
    (folder/'research_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
