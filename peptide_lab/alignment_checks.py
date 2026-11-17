"""Reproducible checks requested by the supplied literature-review framework."""
import json
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from .config import DATA, RESULTS, AA

def high_similarity(query,reference,cutoff=.8):
    # Exact exhaustive search for this metric at the stated cutoff; no k-mer prefilter.
    rows=[]
    for start in range(0,len(query),128):
        batch=query[start:start+128]
        values=process.cdist(batch,reference,scorer=Levenshtein.normalized_similarity,score_cutoff=cutoff,dtype=np.float32,workers=4)
        for sequence,scores in zip(batch,values):
            idx=int(scores.argmax())
            rows.append({'sequence':sequence,'high_similarity_match':reference[idx] if scores[idx]>=cutoff else None,'normalized_edit_similarity':float(scores[idx]) if scores[idx]>=cutoff else None,'has_match_at_or_above_0_8':bool(scores[idx]>=cutoff)})
        if start%1024==0:print('Similarity queries',min(start+128,len(query)),'/',len(query),flush=True)
    return pd.DataFrame(rows)

def main():
    folder=RESULTS/'alignment';folder.mkdir(exist_ok=True)
    tasks=[]
    for task in ['activity','toxicity','vae']:
        training=pd.read_csv(DATA/f'final/{task}_train.csv').sequence.tolist()
        test=pd.read_csv(DATA/f'final/{task}_test.csv').sequence.tolist()
        matches=high_similarity(test,training)
        matches.to_csv(folder/f'{task}_test_training_similarity.csv',index=False)
        tasks.append({'task':task,'test_rows':len(test),'training_rows':len(training),'test_with_training_similarity_ge_0_8':int(matches.has_match_at_or_above_0_8.sum()),'fraction':float(matches.has_match_at_or_above_0_8.mean())})
    generated=pd.read_csv(RESULTS/'ranked_candidates.csv').sequence.tolist()
    known=pd.read_csv(DATA/'processed/grampa_clean_unique_sequences.csv').sequence.tolist()
    high_similarity(generated,known).to_csv(folder/'generated_grampa_similarity.csv',index=False)
    method='Exhaustive normalized Levenshtein similarity >= 0.8 against training sequences (1 - edit_distance/max_length). Scores below cutoff are not stored. This is an edit-similarity diagnostic, not BLAST identity, evolutionary homology, or a clustered split. No weights, thresholds or split assignments are changed from test findings.'
    (folder/'similarity_summary.json').write_text(json.dumps({'tasks':tasks,'method':method},indent=2))
    sources={'training':pd.read_csv(DATA/'final/vae_train.csv').sequence,'generated':pd.read_csv(RESULTS/'vae_raw_generation.csv').sequence}
    composition=pd.DataFrame({'amino_acid':list(AA)})
    for name,seqs in sources.items():
        text=''.join(seqs)
        composition[name]=[text.count(a)/len(text) for a in AA]
    composition.to_csv(folder/'amino_acid_composition.csv',index=False)
    history=pd.read_csv(RESULTS/'vae_history.csv')
    for split in ['train','validation']:
        history[split+'_total_at_epoch_beta']=history[split+'_reconstruction']+history.beta*history[split+'_kl']
    history.to_csv(folder/'vae_objective_history.csv',index=False)
    print('Framework alignment checks complete.',flush=True)

if __name__=='__main__':main()
