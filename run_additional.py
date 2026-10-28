import sys,json,csv,statistics
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score
ROOT=Path(__file__).resolve().parent
P=ROOT
sys.path.insert(0,str(P/'src'))
from models import FEATURES,FlatRuleClassifier,HierarchicalRuleClassifier
D=pd.read_csv(P/'data/car.data',header=None,names=FEATURES+['class'],dtype=str)
seeds=[11,23,37,53,71]
rows=[]
for seed in seeds:
 train,test=train_test_split(D,test_size=.2,random_state=seed,stratify=D['class'])
 for name,model in [('Flat',FlatRuleClassifier()),('Hierarchical',HierarchicalRuleClassifier())]:
  model.fit(train[FEATURES],train['class']); yp=model.predict(test[FEATURES]); yt=test['class']
  rows.append(dict(seed=seed,model=name,accuracy=float(accuracy_score(yt,yp)),macro_f1=float(f1_score(yt,yp,average='macro',zero_division=0)),macro_recall=float(recall_score(yt,yp,average='macro',zero_division=0)),good_recall=float(recall_score(yt,yp,labels=['good'],average='macro',zero_division=0)),vgood_recall=float(recall_score(yt,yp,labels=['vgood'],average='macro',zero_division=0))))
 print('completed seed',seed,flush=True)
with (ROOT/'outputs/repeated_split_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for name in ('Flat','Hierarchical'):
 v=[r for r in rows if r['model']==name];summary[name]={k:{'mean':statistics.mean(r[k] for r in v),'sd':statistics.stdev(r[k] for r in v)} for k in ['accuracy','macro_f1','macro_recall','good_recall','vgood_recall']}
deltas=[next(x['macro_f1'] for x in rows if x['seed']==s and x['model']=='Hierarchical')-next(x['macro_f1'] for x in rows if x['seed']==s and x['model']=='Flat') for s in seeds]
summary['paired_macro_f1_differences']=deltas
(ROOT/'outputs/repeated_split_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
