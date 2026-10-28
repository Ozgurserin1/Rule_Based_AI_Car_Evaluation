import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from models import FEATURES, LABELS, FlatRuleClassifier, HierarchicalRuleClassifier

class RulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.df=pd.read_csv(root/'data'/'car.data',header=None,names=FEATURES+['class'],dtype=str)
        cls.X=cls.df[FEATURES].iloc[:700]
        cls.y=cls.df['class'].iloc[:700]
    def test_flat_predicts_allowed_classes_for_every_record(self):
        m=FlatRuleClassifier().fit(self.X,self.y)
        p=m.predict(self.X)
        self.assertEqual(len(p),len(self.X))
        self.assertTrue(set(p).issubset(LABELS))
    def test_hierarchical_predicts_allowed_classes_for_every_record(self):
        m=HierarchicalRuleClassifier().fit(self.X,self.y)
        p=m.predict(self.X)
        self.assertEqual(len(p),len(self.X))
        self.assertTrue(set(p).issubset(LABELS))
    def test_hierarchical_intermediate_explanation(self):
        m=HierarchicalRuleClassifier().fit(self.X,self.y)
        t=m.explain({'buying':'low','maint':'low','doors':'4','persons':'4','lug_boot':'big','safety':'high'})
        self.assertEqual(t['PRICE'],'low')
        self.assertIn('TECH',t)
    def test_flat_deterministic(self):
        a=FlatRuleClassifier().fit(self.X,self.y)
        b=FlatRuleClassifier().fit(self.X,self.y)
        self.assertEqual(a.predict(self.X).tolist(),b.predict(self.X).tolist())
if __name__ == '__main__': unittest.main()
