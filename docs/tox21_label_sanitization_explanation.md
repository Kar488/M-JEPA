# Tox21 label sanitization context

During the logged CI run the Tox21 pipeline loaded labels that included values outside the binary {0,1} set. The training loop treated these as-is, so rows with `-1` or other non-binary entries leaked into splits, skewing class balance and confusing the classifier. This manifested as unstable training and very low ROC-AUC (≈0.44) despite reasonable hyperparameters.

The fix introduces a `_sanitize_binary_labels` helper in `experiments/case_study.py` that:

1. Casts labels to integers, dropping any rows that are not 0/1.
2. Reports how many rows were removed so diagnostics are transparent.
3. Passes the cleaned labels downstream for splits, loaders, and calibration.

With invalid rows removed, class weights and calibration are computed from consistent data, which should lead to more reliable training dynamics and better AUC.
