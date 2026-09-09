# R1 revision results

This directory is reserved for **new reviewer-response experiments only**.

Rules:

- Never overwrite V1.0 `results_reproduced/` or `results_reference/`.
- Cloud-development outputs are provisional and must not be treated as paper-final evidence.
- Final R1 result CSV/JSON files should be generated on the user's local machine from the SHA-matched 539-D frozen archive using the `revision/r1-202609` branch.
- Neural baseline final results require five folds, the three locked seeds, validation-only hyperparameter selection and complete logs.
- Any result used in the revised manuscript or response letter must have a documented command/config and must preserve the five independent two-station deployment-group boundary for daily inference.
