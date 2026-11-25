"""Frozen official ESM-2 residue embeddings with independently trained CNN heads.

This is an ESM-2/CNN adaptation, not an exact UniDL4BioPep reproduction.
Mean pooling excludes CLS, EOS and padding. No task labels enter the backbone.
"""
import json
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .config import ROOT, DATA, RESULTS
from .training import metrics, seed_everything

BACKBONE = ROOT / 'Models/esm2_backbone'
OUTPUT = RESULTS / 'esm2'
CACHE = DATA / 'esm2'

class ESMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
                                     nn.Flatten(), nn.Linear(32*159, 64), nn.ReLU(),
                                     nn.Dropout(.15), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x.unsqueeze(1)).squeeze(1)

class ESMEmbedder:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(str(BACKBONE), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(BACKBONE), local_files_only=True, add_pooling_layer=False).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, sequences, progress=False):
        # Length sorting reduces padding, and therefore transformer CPU cost.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        out = np.empty((len(sequences), 320), dtype=np.float32)
        for start in range(0, len(order), 32):
            indices = order[start:start+32]
            batch = self.tokenizer([sequences[i] for i in indices], return_tensors='pt', padding=True)
            states = self.model(**batch).last_hidden_state
            mask = batch['attention_mask'].bool()
            for token_id in self.tokenizer.all_special_ids:
                mask &= batch['input_ids'].ne(token_id)
            pooled = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            out[indices] = pooled.numpy()
            if progress and (start % 1024 == 0 or start + 32 >= len(order)):
                print(f'ESM embeddings {min(start+32,len(order))}/{len(order)}', flush=True)
        return out

def embed_dataset():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(DATA / 'final' / f'{task}_{split}.csv') for task in ['activity','toxicity'] for split in ['train','validation','test']]
    sequences = sorted(set(pd.concat(frames).sequence))
    fingerprint = hashlib.sha256(('\n'.join(sequences) + (BACKBONE/'source.json').read_text()).encode()).hexdigest()
    meta_path = CACHE/'manifest.json'
    if meta_path.exists() and json.loads(meta_path.read_text()).get('fingerprint') == fingerprint and (CACHE/'embeddings.npy').exists():
        print('Reusing verified sequence-keyed embedding cache.', flush=True)
        values = np.load(CACHE/'embeddings.npy')
        if values.shape != (len(sequences),320):
            raise ValueError('Embedding cache shape mismatch.')
    else:
        values = ESMEmbedder().encode(sequences, progress=True)
        np.save(CACHE/'embeddings.npy', values)
        pd.DataFrame({'sequence':sequences}).to_csv(CACHE/'sequences.csv',index=False)
        meta_path.write_text(json.dumps({'fingerprint':fingerprint,'rows':len(sequences),'dimension':320,'pooling':'Mean of residue tokens from last hidden state; special/padding excluded.','backbone':json.loads((BACKBONE/'source.json').read_text())},indent=2))
    return sequences, values

@torch.inference_mode()
def scores(model, x):
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.as_tensor(x[i:i+256],dtype=torch.float32))).numpy() for i in range(0,len(x),256)])

def train_head(task, sequences, embeddings):
    import copy
    seed_everything()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    model_folder = ROOT/'Models/esm2_heads'
    model_folder.mkdir(parents=True,exist_ok=True)
    lookup={s:i for i,s in enumerate(sequences)}
    frames={part:pd.read_csv(DATA/'final'/f'{task}_{part}.csv') for part in ['train','validation','test']}
    x={part:embeddings[[lookup[s] for s in frame.sequence]] for part,frame in frames.items()}
    scaler=StandardScaler().fit(x['train'])
    x={part:scaler.transform(value).astype(np.float32) for part,value in x.items()}
    y={part:frame.label.to_numpy(dtype=np.float32) for part,frame in frames.items()}
    model=ESMHead()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y['train']==0).sum()/(y['train']==1).sum())))
    loader=DataLoader(TensorDataset(torch.as_tensor(x['train']),torch.as_tensor(y['train'])),batch_size=128,shuffle=True)
    best_auc,best,stale,best_epoch=-1,None,0,0
    history=[]
    for epoch in range(1,31):
        model.train();total=0.
        for bx,by in loader:
            optimizer.zero_grad();loss=loss_fn(model(bx),by);loss.backward();optimizer.step()
            total+=loss.item()*len(by)
        val=scores(model,x['validation']);auc=roc_auc_score(y['validation'],val)
        history.append({'epoch':epoch,'training_loss':total/len(y['train']),'validation_auc':auc})
        print(task,'ESM epoch',epoch,'validation AUC',round(auc,5),flush=True)
        if auc>best_auc+.0001:
            best_auc,best,best_epoch,stale=auc,copy.deepcopy(model.state_dict()),epoch,0
        else:stale+=1
        if stale>=6:break
    model.load_state_dict(best)
    validation_frame=frames['validation'].copy()
    validation_frame['esm2_score']=scores(model,x['validation'])
    validation_frame.to_csv(OUTPUT/f'{task}_validation_predictions.csv',index=False)
    # Compare validation scores before examining this head's test outcomes.
    baseline=json.loads((RESULTS/f'{task}_metrics.json').read_text())
    baseline_auc=baseline['validation_auc'][baseline['selected_model']]
    recommended='esm2_cnn' if best_auc>baseline_auc else 'compact_baseline'
    torch.save(best,model_folder/f'{task}.pt')
    joblib.dump(scaler,model_folder/f'{task}_scaler.joblib')
    pred=scores(model,x['test'])
    frame=frames['test'].copy();frame['esm2_score']=pred;frame.to_csv(OUTPUT/f'{task}_test_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(OUTPUT/f'{task}_history.csv',index=False)
    report={'task':task,'architecture':'Frozen esm2_t6_8M_UR50D -> residue mean pooling (320) -> train-only StandardScaler -> Conv1D(32,k=3), MaxPool, Dense(64), dropout(.15), sigmoid',
            'best_epoch':best_epoch,'validation_auc':best_auc,'baseline_validation_auc':baseline_auc,'validation_recommendation':recommended,
            'test':metrics(y['test'],pred),'threshold':.5,'backbone_revision':json.loads((BACKBONE/'source.json').read_text())['revision'],
            'protocol':'Same exact splits as compact baseline. Checkpoint and recommendation selected using validation only. Frozen pretrained backbone; held-out labels not used in embedding extraction. Existing test data was previously inspected, so this is a documented follow-up comparison, not a new untouched external benchmark.'}
    (OUTPUT/f'{task}_metrics.json').write_text(json.dumps(report,indent=2))
    print(task,'ESM final',json.dumps(report['test']),flush=True)

class ESMPredictor:
    def __init__(self):
        self.embedder=ESMEmbedder()
        self.heads={}
        for task in ['activity','toxicity']:
            head=ESMHead()
            head.load_state_dict(torch.load(ROOT/f'Models/esm2_heads/{task}.pt',map_location='cpu',weights_only=True))
            self.heads[task]=(head.eval(),joblib.load(ROOT/f'Models/esm2_heads/{task}_scaler.joblib'))
        self.lookup={}
        cache_path=CACHE/'sequences.csv'
        if cache_path.exists() and (CACHE/'embeddings.npy').exists():
            manifest=json.loads((CACHE/'manifest.json').read_text())
            if manifest['backbone']['revision'] != json.loads((BACKBONE/'source.json').read_text())['revision']:
                raise ValueError('Embedding cache backbone mismatch; regenerate embeddings.')
            for name,expected in manifest.get('cache_files_sha256',{}).items():
                if hashlib.sha256((CACHE/name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Embedding cache checksum mismatch: '+name)
            self.cached=np.load(CACHE/'embeddings.npy',mmap_mode='r')
            self.lookup={s:i for i,s in enumerate(pd.read_csv(cache_path).sequence)}
            if self.cached.shape != (len(self.lookup),320):
                raise ValueError('Embedding cache sequence/shape mismatch.')

    def predict(self,sequences):
        matrix=np.empty((len(sequences),320),dtype=np.float32)
        missing=[i for i,s in enumerate(sequences) if s not in self.lookup]
        for i,s in enumerate(sequences):
            if s in self.lookup:matrix[i]=self.cached[self.lookup[s]]
        if missing:matrix[missing]=self.embedder.encode([sequences[i] for i in missing])
        return {task:scores(head,scaler.transform(matrix).astype(np.float32)) for task,(head,scaler) in self.heads.items()}

def main():
    started=time.time()
    sequences,embeddings=embed_dataset()
    for task in ['activity','toxicity']:train_head(task,sequences,embeddings)
    print('Finished ESM-2 comparison in',round(time.time()-started,1),'seconds',flush=True)

if __name__=='__main__':main()
