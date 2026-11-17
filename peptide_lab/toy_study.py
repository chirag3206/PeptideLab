"""New controlled toy study; never overwrites deployed or historical weights."""
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from .config import DATA, RESULTS
from .networks import PeptideVAE
from .features import properties
from .training import seed_everything, vae_arrays
from .preprocessing import prepare
from .evaluation import decode

SEEDS=[42,101,123,256,512,777,999,2024,3141,4096]

def experiment(pool,seed,latent=24,dropout=.2,epochs=30):
    seed_everything(seed)
    sample=pd.Series(pool).sample(n=200,random_state=seed).tolist()
    train,held=train_test_split(sample,test_size=.2,random_state=seed)
    val,test=train_test_split(held,test_size=.5,random_state=seed)
    folder=RESULTS/'extended/toy_runs'/f'seed_{seed}_latent_{latent}_dropout_{dropout}'
    folder.mkdir(parents=True,exist_ok=True)
    config={'seed':seed,'latent':latent,'word_dropout':dropout,'epochs':epochs,'batch_size':32,'embedding':32,'hidden':96,'beta_max':.02,'warmup_epochs':10,'length_range':[5,40],'subset_size':200,'splits':{'train':160,'validation':20,'test':20}}
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    for name,sequences in [('train',train),('validation',val),('test',test)]:
        pd.DataFrame({'sequence':sequences}).to_csv(folder/f'{name}.csv',index=False)
    x,y=vae_arrays(train,length=41);vx,vy=vae_arrays(val,length=41)
    model=PeptideVAE(latent=latent,word_dropout=dropout)
    loader=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    best_loss,best=1e9,None
    history=[]
    for epoch in range(1,epochs+1):
        beta=.02*min(1,epoch/10)
        model.train();total,count=0.,0
        for bx,by in loader:
            logits,mu,lv=model(bx)
            n=int(by.ne(0).sum())
            recon=torch.nn.functional.cross_entropy(logits.flatten(0,1),by.flatten(),ignore_index=0,reduction='sum')
            kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
            opt.zero_grad();(recon/n+beta*kl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();total+=recon.item();count+=n
        model.eval()
        with torch.no_grad():
            logits,mu,lv=model(vx)
            recon=torch.nn.functional.cross_entropy(logits.flatten(0,1),vy.flatten(),ignore_index=0,reduction='sum')/vy.ne(0).sum()
            kl=-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean()
            objective=float(recon+.02*kl)
        history.append({'epoch':epoch,'train_reconstruction':total/count,'validation_reconstruction':float(recon),'validation_kl':float(kl),'beta':beta})
        if objective<best_loss:
            best_loss,best=objective,copy.deepcopy(model.state_dict())
    model.load_state_dict(best);model.eval()
    tx,ty=vae_arrays(test,length=41)
    with torch.no_grad():
        logits,mu,lv=model(tx);pred=logits.argmax(-1);mask=ty.ne(0)
        recon=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),ty.flatten(),ignore_index=0,reduction='sum')/mask.sum())
        kl=float(-.5*(1+lv-mu.square()-lv.exp()).sum(1).mean())
    generated,ended=model.sample(100,seed=seed,temperature=.9,max_length=40)
    known=set(pool)
    row={'seed':seed,'latent':latent,'word_dropout':dropout,'test_cross_entropy':recon,'test_kl':kl,'token_accuracy':float(((pred==ty)&mask).sum()/mask.sum()),'teacher_forced_exact_match':float(np.mean([decode(a)==b for a,b in zip(pred.tolist(),test)])),
         'uniqueness':len(set(generated))/len(generated),'exact_novelty_vs_5_40_pool':sum(s not in known for s in set(generated))/len(set(generated)),
         'natural_end_rate':np.mean(ended),'mean_length':np.mean(list(map(len,generated))),'mean_entropy':np.mean([properties(s)['entropy'] for s in generated])}
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
    pd.DataFrame({'sequence':generated,'ended_naturally':ended}).to_csv(folder/'generated.csv',index=False)
    (folder/'metrics.json').write_text(json.dumps(row,indent=2))
    torch.save(best,folder/'weights.pt')
    print('Finished',folder.name,'test CE',round(recon,3),flush=True)
    return row

def main():
    prepare()
    pool=pd.read_csv(DATA/'final/grampa_5_40.csv').sequence.tolist()
    configurations=[(s,24,.2) for s in SEEDS]+[(s,l,.2) for s in SEEDS[:3] for l in [16,32,64]]+[(s,24,0.) for s in SEEDS[:3]]
    rows=[]
    for seed,latent,dropout in configurations:
        path=RESULTS/'extended/toy_runs'/f'seed_{seed}_latent_{latent}_dropout_{dropout}'/'metrics.json'
        # Completed run folders are immutable and resumable for this fixed protocol.
        row=json.loads(path.read_text()) if path.exists() else experiment(pool,seed,latent,dropout)
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS/'extended/toy_study_runs.csv',index=False)
    frame=pd.DataFrame(rows)
    numeric=['test_cross_entropy','test_kl','token_accuracy','teacher_forced_exact_match','uniqueness','natural_end_rate','mean_entropy']
    summary=[]
    for (latent,dropout),group in frame.groupby(['latent','word_dropout']):
        # Report matched 3-seed default separately from the 10-seed robustness result.
        variants=[('all_available_seeds',group)]
        if len(group)>3:
            variants.append(('matched_first_three_seeds',group[group.seed.isin(SEEDS[:3])]))
        for scope,g in variants:
            for metric in numeric:
                values=g[metric].to_numpy()
                rng=np.random.default_rng(42)
                boot=rng.choice(values,(2000,len(values)),replace=True).mean(1)
                low,high=np.quantile(boot,[.025,.975])
                summary.append({'latent':latent,'word_dropout':dropout,'seed_scope':scope,'runs':len(g),'metric':metric,'mean':values.mean(),'std':values.std(ddof=1),'bootstrap_mean_lower95':low,'bootstrap_mean_upper95':high})
    pd.DataFrame(summary).to_csv(RESULTS/'extended/toy_study_summary.csv',index=False)
    print('Completed',len(rows),'controlled runs. Bootstrap intervals with 3 seeds are highly uncertain.',flush=True)

if __name__=='__main__':
    main()
