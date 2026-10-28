# Car Evaluation — Rule-Based AI Research (CPU5006)

Two custom Python rule-based classifiers (flat sequential covering; hierarchical symbolic aggregation), evaluated on the UCI Car Evaluation Dataset.

## Research integrity
The final 346-row holdout was tested in Step 6; it is no longer untouched. **Do not retune models against its results.** The frozen methodology and original source hashes are recorded in `METHODOLOGY_FROZEN.json`. AI-assistance attribution comments were added to `src/*.py` after the final evaluation; `CODE_ANNOTATION_AUDIT.json` records the revised hashes. No executable statements were changed.

## Setup and replication
Requires Python 3.10+; from repository root:
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_experiment.py
```
To independently reproduce the **already published** final test (this overwrites `outputs/FINAL_holdout_results.json` with equivalent results):
```bash
python src/run_experiment.py --final-test
```
Comparative tables in `outputs/`; figures in `figures/`; Step 6 methods in `STEP6_REPORT.md`.

## Research limitations
Different rule-construction strategies and hand-designed intermediate features confound a pure architectural comparison. Rare classes are small, and human interpretability was not tested. No real vehicle performance claims are made.

## Data attribution
Bohanec, M. (1988) *Car Evaluation*. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48. Data licensed CC BY 4.0.



## Exploratory robustness analyses (post-hoc)
Run the two additional exploratory experiment scripts from the repository root:
```bash
python run_additional.py
python run_controlled.py
```
These use stratified 80/20 train-test splits with seeds 11, 23, 37, 53, 71. The same records are reused across splits, so these are not independent external validations. The original fixed-seed 42 holdout is preserved as the initial reported analysis.

`run_controlled.py` applies the **same flat-rule learner** to (a) six raw inputs, and (b) the intermediate PRICE, COMFORT and TECH states of the hierarchical model. To reuse the original six-feature learner interface, the three states are passed via the first three feature columns and the remaining three columns are constant. This contrast changes representation/information content and does not isolate the effect of hierarchy.

### Important interpretation
The additional checks were designed after inspecting the initial holdout results. Treat the findings as exploratory, do not claim confirmatory statistical significance or a new untouched test set, and do not interpret rule traces as human interpretability evidence.
