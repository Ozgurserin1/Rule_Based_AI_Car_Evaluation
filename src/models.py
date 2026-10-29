"""Two transparent, exclusively rule-based classifiers for Car Evaluation."""
from dataclasses import dataclass
from itertools import combinations, product
from collections import Counter
import numpy as np
import pandas as pd

FEATURES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
LABELS = ['unacc', 'acc', 'good', 'vgood']

@dataclass(frozen=True)
class Rule:
    conditions: tuple
    prediction: str
    coverage: int
    precision: float
    def matches(self, record):
        return all(str(record[k]) == v for k, v in self.conditions)

class FlatRuleClassifier:
    """Sequential covering with candidate conjunctions of at most 3 conditions.

    Every selection is based on training labels only. An ordered decision list
    resolves overlaps by first match; fallback is the training-majority class.
    """
    def __init__(self, max_conditions=3, min_support=8, min_purity=0.72, max_rules=65):
        self.max_conditions = max_conditions
        self.min_support = min_support
        self.min_purity = min_purity
        self.max_rules = max_rules

    def fit(self, X, y):
        X = X[FEATURES].reset_index(drop=True).astype(str)
        y = pd.Series(y).reset_index(drop=True).astype(str).to_numpy()
        self.fallback = Counter(y).most_common(1)[0][0]
        n = len(y)
        candidates = []
        for depth in range(1, self.max_conditions+1):
            for cols in combinations(FEATURES, depth):
                for values in product(*[sorted(X[col].unique()) for col in cols]):
                    mask = np.ones(n, dtype=bool)
                    conditions = tuple(zip(cols, values))
                    for col, value in conditions:
                        mask &= X[col].to_numpy() == value
                    if mask.sum() >= self.min_support:
                        candidates.append((conditions, mask))
        remaining = np.ones(n, dtype=bool)
        self.rules = []
        for _ in range(self.max_rules):
            best = None
            for conditions, base_mask in candidates:
                mask = base_mask & remaining
                support = int(mask.sum())
                if support < self.min_support:
                    continue
                counts = Counter(y[mask])
                label, correct = sorted(counts.items(), key=lambda kv: (-kv[1], LABELS.index(kv[0])))[0]
                purity = correct/support
                if purity < self.min_purity:
                    continue
                # prioritize correct coverage, precision and simplicity
                score = (correct - 0.7*(support-correct)) / (1 + .12*(len(conditions)-1))
                if best is None or score > best[0]:
                    best = (score, conditions, mask, label, support, purity)
            if best is None:
                break
            _, conds, mask, label, support, purity = best
            self.rules.append(Rule(conds, label, support, round(purity,4)))
            remaining &= ~mask
            if remaining.sum() == 0:
                break
        self.uncovered_training_rows = int(remaining.sum())
        return self

    def predict_one(self, record):
        for rule in self.rules:
            if rule.matches(record):
                return rule.prediction
        return self.fallback

    def predict(self, X):
        return np.array([self.predict_one(record) for record in X[FEATURES].astype(str).to_dict('records')])

    def explain(self, record):
        for i, rule in enumerate(self.rules, 1):
            if rule.matches(record):
                return {'prediction':rule.prediction,'rule_number':i,'conditions':dict(rule.conditions)}
        return {'prediction':self.fallback,'rule_number':None,'conditions':'default'}

class HierarchicalRuleClassifier:
    """Symbolic, domain-defined intermediate rules and supervised terminal lookup.

    Intermediate states are researcher-designed approximations, NOT the original
    dataset generator's unprovided expert rules. Only the terminal lookup is
    fitted from labelled training examples.
    """
    def _intermediates(self, record):
        costs = ['low','med','high','vhigh']
        cost_idx = max(costs.index(record['buying']), costs.index(record['maint']))
        price = costs[cost_idx]
        if record['persons'] == '2':
            comfort = 'restricted'
        elif record['persons'] == 'more' and record['lug_boot'] in ('med','big') and record['doors'] in ('4','5more'):
            comfort = 'spacious'
        elif record['lug_boot'] == 'small' or record['doors'] == '2':
            comfort = 'basic'
        else:
            comfort = 'standard'
        if record['safety'] == 'low' or comfort == 'restricted':
            tech = 'unsuitable'
        elif record['safety'] == 'high' and comfort == 'spacious':
            tech = 'excellent'
        elif record['safety'] == 'high' and comfort in ('standard','basic'):
            tech = 'good'
        elif record['safety'] == 'med' and comfort in ('spacious','standard'):
            tech = 'adequate'
        else:
            tech = 'limited'
        return price, comfort, tech

    def fit(self, X, y):
        X = X[FEATURES].reset_index(drop=True).astype(str)
        y = pd.Series(y).reset_index(drop=True).astype(str)
        table = {}
        for rec, cls in zip(X.to_dict('records'), y):
            price, _, tech = self._intermediates(rec)
            table.setdefault((price,tech), Counter())[cls] += 1
        self.lookup = {k: sorted(counts.items(), key=lambda kv:(-kv[1],LABELS.index(kv[0])))[0][0] for k,counts in table.items()}
        self.fallback = Counter(y).most_common(1)[0][0]
        self.training_groups = {k:dict(v) for k,v in table.items()}
        return self

    def predict_one(self, record):
        price, comfort, tech = self._intermediates(record)
        return self.lookup.get((price,tech),self.fallback)

    def predict(self, X):
        return np.array([self.predict_one(rec) for rec in X[FEATURES].astype(str).to_dict('records')])

    def explain(self, record):
        price, comfort, tech = self._intermediates(record)
        return {'PRICE':price,'COMFORT':comfort,'TECH':tech,'prediction':self.lookup.get((price,tech),self.fallback),'final_rule':f'IF PRICE={price} AND TECH={tech}'}
