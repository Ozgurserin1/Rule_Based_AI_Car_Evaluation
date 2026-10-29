# CPU5006 Python Project — Verification and Research Status

**Dataset:** The user's latest uploaded Car Evaluation Dataset ZIP; `data/car.data` matches the original uploaded data bytes (SHA-256: b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83).

**Automated tests:** 4 of 4 passed.

**Original final held-out evaluation:** 1,036 training, 346 validation, 346 final test; seed 42. Final results independently rerun from source.
- Flat: accuracy 0.87861, macro-F1 0.73229, 304/346 correct.
- Hierarchical: accuracy 0.88728, macro-F1 0.76515, 307/346 correct.

**Exploratory robustness study:** Five stratified 80/20 splits (seeds 11, 23, 37, 53, 71). Mean accuracy flat 0.88497, hierarchical 0.86069; mean macro-F1 flat 0.76150, hierarchical 0.72486. The flat model wins macro-F1 on 4 of the 5 additional splits.

**Exploratory representation study:** Same FlatRuleClassifier on the raw six inputs vs transformed PRICE/COMFORT/TECH representation. Mean macro-F1 0.76150 vs 0.65784. The transformed representation has fewer distinct input features, so this does **not** isolate the causal effect of hierarchy.

**Research limitations:** The 346-row original holdout is previously exposed and must not be described as untouched in future research. The extra splits reuse the same dataset, are exploratory and post-hoc, and do not provide a new independent holdout. No inferential p-values or claims of user-tested interpretability are warranted. The hierarchical model includes handcrafted intermediate rules; the two main systems differ in architecture and rule-generation technique.

**How to reproduce:**
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_experiment.py --final-test  # Re-runs and overwrites the saved final holdout results; do not run for routine verification
python run_additional.py
python run_controlled.py
```

**Publication status (29 October 2026):** The public GitHub repository contains the original 28-file commit `52c2471` and the subsequent documentation update commit `894d688` (`Update project documentation and verification report`). The update was successfully pushed to `main`. The VS Code `git status` check confirmed that the local branch was up to date with `origin/main` and that the working tree was clean. Repository: https://github.com/Ozgurserin1/Rule_Based_AI_Car_Evaluation. Overleaf conversion remains pending.
