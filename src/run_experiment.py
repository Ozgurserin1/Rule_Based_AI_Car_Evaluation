# AI assistance disclosure: ChatGPT supported initial code development, review and documentation.
# The research author must inspect, understand and verify these implementations and results.
"""Execute reproducible development-set checks; keep final holdout sealed by default."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models import FEATURES, LABELS, FlatRuleClassifier, HierarchicalRuleClassifier

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    file = ROOT/'data'/'car.data'
    df = pd.read_csv(file, header=None, names=FEATURES+['class'], dtype=str)
    assert df.shape == (1728,7), f'Unexpected dataset dimensions: {df.shape}'
    assert not df.isna().any().any()
    assert not df.duplicated(subset=FEATURES).any()
    assert set(df['class']) == set(LABELS)
    return df

def report(name, model, X, y):
    yp = model.predict(X)
    return dict(name=name,accuracy=round(accuracy_score(y,yp),5),macro_f1=round(f1_score(y,yp,labels=LABELS,average='macro',zero_division=0),5),classification_report=classification_report(y,yp,labels=LABELS,output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(y,yp,labels=LABELS).tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-test',action='store_true',help='UNSEAL final holdout: only use once all design decisions are frozen (Step 6).')
    args = parser.parse_args()
    df = load_data()
    # Fixed external holdout not used for rule construction or tuning.
    dev, test = train_test_split(df,test_size=.2,random_state=42,stratify=df['class'])
    train, val = train_test_split(dev,test_size=.25,random_state=42,stratify=dev['class'])
    X_train, y_train = train[FEATURES],train['class']
    X_val, y_val = val[FEATURES],val['class']
    models = [FlatRuleClassifier().fit(X_train,y_train),HierarchicalRuleClassifier().fit(X_train,y_train)]
    print('Splits:', {'train':len(train),'validation':len(val),'sealed_holdout':len(test)})
    for model in models:
        predictions = model.predict(X_val)
        assert len(predictions)==len(X_val) and set(predictions).issubset(LABELS)
        print('\n',type(model).__name__,': successful prediction coverage',len(predictions),'/',len(X_val))
        if isinstance(model,FlatRuleClassifier):
            print('Learned rules:',len(model.rules),'train rows requiring fallback:',model.uncovered_training_rows)
            for rule in model.rules[:5]: print(' ',rule)
        else: print('Learned final lookup groups:',len(model.lookup))
        print('Sample explanation:',model.explain(X_val.iloc[0].to_dict()))
    if args.final_test:
        # Refit on all development data only after selection and freezing settings.
        models = [FlatRuleClassifier().fit(dev[FEATURES],dev['class']),HierarchicalRuleClassifier().fit(dev[FEATURES],dev['class'])]
        results=[report(type(m).__name__,m,test[FEATURES],test['class']) for m in models]
        dest=ROOT/'outputs'/'FINAL_holdout_results.json'
        dest.write_text(json.dumps({'WARNING':'Final holdout exposed: do not retune based on results','labels':LABELS,'results':results},indent=2))
        print('Final results written to:',dest)
    else:
        results=[report(type(m).__name__,m,X_val,y_val) for m in models]
        dest=ROOT/'outputs'/'development_validation.json'
        dest.write_text(json.dumps({'NOTE':'Development validation only: NOT final scientific findings','labels':LABELS,'results':results},indent=2))
        print('Development validation written to:',dest)
    for item in results: print(item['name'],'accuracy:',item['accuracy'],'macro-F1:',item['macro_f1'])

if __name__ == '__main__': main()
