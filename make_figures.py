"""Create publication-ready figures/tables from the frozen final test JSON only."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'outputs'/'FINAL_holdout_results.json').read_text())
labels=D['labels']; results=D['results']; names=['Flat rules','Hierarchical rules']
rows=[]
for r,n in zip(results,names):
    report=r['classification_report']
    rows.append({'Algorithm':n,'Test N':346,'Accuracy':r['accuracy'], 'Macro precision':report['macro avg']['precision'],'Macro recall':report['macro avg']['recall'],'Macro F1':r['macro_f1'], 'Correct':int(np.trace(np.array(r['confusion_matrix'])))})
with (ROOT/'outputs'/'final_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with (ROOT/'outputs'/'final_per_class.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['Algorithm','Class','Support','Precision','Recall','F1'])
 for r,n in zip(results,names):
  for c in labels:
   met=r['classification_report'][c];w.writerow([n,c,int(met['support']),met['precision'],met['recall'],met['f1-score']])
metrics=['Accuracy','Macro precision','Macro recall','Macro F1']
xs=np.arange(len(metrics)); fig,ax=plt.subplots(figsize=(9,5.2))
for i,row in enumerate(rows):
 values=[row[k]*100 for k in metrics]
 bars=ax.bar(xs+(i-0.5)*0.35,values,0.35,label=row['Algorithm'])
 ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax.set_xticks(xs,metrics);ax.set_ylabel('Score (%)');ax.set_ylim(0,105);ax.set_title('Figure 1. Held-out performance (n = 346)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures'/'figure_1_model_comparison.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.2));xs=np.arange(len(labels))
for i,(r,n) in enumerate(zip(results,names)):
 vals=[r['classification_report'][c]['recall']*100 for c in labels]
 bars=ax.bar(xs+(i-.5)*.35,vals,.35,label=n);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax.set_xticks(xs,labels);ax.set_ylim(0,105);ax.set_ylabel('Recall (%)');ax.set_title('Figure 2. Per-class recall on the held-out test set');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures'/'figure_2_per_class_recall.png',dpi=220);plt.close(fig)
for ix,(r,n) in enumerate(zip(results,names),start=3):
 cm=np.asarray(r['confusion_matrix']);fig,ax=plt.subplots(figsize=(6.4,5.4));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(4),labels);ax.set_yticks(range(4),labels);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'Figure {ix}. {n} confusion matrix (n = 346)')
 for i in range(4):
  for j in range(4):
   ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>(cm.max()/2) else 'black')
 fig.colorbar(im,ax=ax,shrink=.8,label='Number of records');fig.tight_layout();fig.savefig(ROOT/'figures'/f'figure_{ix}_confusion_matrix.png',dpi=220);plt.close(fig)
print('Created summary tables and 4 figures')
