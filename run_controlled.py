import sys,json,csv,statistics
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent;P=ROOT;sys.path.insert(0,str(P/'src'))
import models
cols=models.FEATURES.copy()
D=pd.read_csv(P/'data/car.data',header=None,names=cols+['class'],dtype=str)
def transform(X):
 h=models.HierarchicalRuleClassifier()
 xx=X.copy()
 vals=[h._intermediates(rec) for rec in X[cols].to_dict('records')]
 for c,j in zip(cols,range(6)):
  if j<3: xx[c]=[v[j] for v in vals]
  else: xx[c]='constant'
 return xx
seeds=[11,23,37,53,71];rows=[]
for seed in seeds:
 tr,te=train_test_split(D,test_size=.2,random_state=seed,stratify=D['class'])
 for representation in ['Original six inputs','PRICE COMFORT TECH inputs']:
  if representation=='Original six inputs': a,b=tr[cols],te[cols]
  else:a,b=transform(tr[cols]),transform(te[cols])
  m=models.FlatRuleClassifier().fit(a,tr['class']);yp=m.predict(b)
  rows.append(dict(seed=seed,representation=representation,accuracy=float(accuracy_score(te['class'],yp)),macro_f1=float(f1_score(te['class'],yp,average='macro',zero_division=0)),rules=len(m.rules)))
 print('controlled seed',seed,flush=True)
with (ROOT/'outputs/controlled_representation_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps({label:{'macro_f1_mean':statistics.mean(r['macro_f1'] for r in rows if r['representation']==label),'accuracy_mean':statistics.mean(r['accuracy'] for r in rows if r['representation']==label),'rules_mean':statistics.mean(r['rules'] for r in rows if r['representation']==label)} for label in ['Original six inputs','PRICE COMFORT TECH inputs']},indent=2))
