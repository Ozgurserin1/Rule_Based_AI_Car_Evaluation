# Step 6 — Final Experiment Results and Methodology

**Status:** completed final held-out evaluation, 27 October 2026. No algorithm settings were modified after viewing the final test results.

## Experiment design

- Car Evaluation: 1,728 distinct complete labelled records, 6 categorical inputs, 4 output classes.
- Stratified 80% development / 20% test with `random_state=42`. Development was further divided into 75% rule learning / 25% validation, providing 1,036 training, 346 validation and 346 test examples.
- All algorithm settings and `src/models.py` code were frozen **before** final evaluation in `METHODOLOGY_FROZEN.json`, including SHA-256 hashes.
- Flat rule induction settings: maximum 3 conditions; support >=8; purity >=0.72; max 65 ordered rules; fallback training majority. These are prior Step 5 defaults, *not* selected using held-out data.
- Hierarchical model: researcher-authored intermediate PRICE, COMFORT, TECH mappings; majority-class terminal lookup learned from labelled development examples. These are **not** the original unprovided DEX rule tables.
- Development models were refitted on the complete 80% development split and each evaluated once on the same test records.
- Primary metric: macro-F1; secondary: accuracy, macro-precision, macro-recall, per-class precision/recall/F1, confusion matrices.

## Final held-out test results (n=346)

| Measure | Flat | Hierarchical |
|---|---:|---:|
| Accuracy | 0.87861 | 0.88728 |
| Macro-F1 | 0.73229 | 0.76515 |
| Macro-precision | 0.86276 | 0.75729 |
| Macro-recall | 0.69454 | 0.78234 |
| Correctly classified | 304/346 | 307/346 |

Class order for confusion matrices: unacc, acc, good, vgood; rows actual and columns predicted. Per-class metrics are in `outputs/final_per_class.csv` and the detailed matrices are stored in `outputs/FINAL_holdout_results.json`.

## Interpretation

The hierarchical model has a macro-F1 advantage of 0.03286 and a numerical accuracy advantage of 0.87 percentage points (three more correct predictions). The flat approach achieves higher macro-precision, while the hierarchy has higher macro-recall. In particular the hierarchical model detects 7/13 very-good cases against the flat model's 4/13. The minority categories comprise few test records, so these comparisons are unstable.

**Important validity limitation:** Since the systems differ in more than architecture (automatic direct-rule induction versus hand-designed intermediate groups and a supervised terminal lookup), no causal inference can isolate hierarchy as the reason for a difference. Training, validation and test samples originate from one synthetic, full-combination car-evaluation decision-model dataset; deployment-world generalisability is not established. Neither a significance test nor multiple-seed uncertainty analysis was prespecified; do not claim statistically significant improvement.

## Figures and raw results

- Figure 1: Overall performance comparison (`figures/figure_1_model_comparison.png`)
- Figure 2: Per-class recall (`figures/figure_2_per_class_recall.png`)
- Figure 3: Flat classifier confusion matrix (`figures/figure_3_confusion_matrix.png`)
- Figure 4: Hierarchical classifier confusion matrix (`figures/figure_4_confusion_matrix.png`)
- Machine-readable output: `outputs/FINAL_holdout_results.json`, `outputs/final_summary.csv`, `outputs/final_per_class.csv`.

## Reproduction

From the project folder:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_experiment.py --final-test
python make_figures.py
```

`--final-test` re-evaluates the already exposed fixed holdout for reproducibility; it is **not** a new independent test. Python libraries in this execution: numpy 2.3.5; pandas 2.2.3; scikit-learn 1.8.0.

## Required attribution / submission

Before submitting, review and accurately describe ChatGPT assistance in the report and in code comments according to the brief. Upload the Python project to a public GitHub repository and place its actual URL near the beginning of the scientific paper. No GitHub submission or paper submission has been performed here.
