# Project Mission

Build a reproducible educational machine-learning project using
the Multicultural Adolescents Panel Study (MAPS) first cohort.

Research goal:

Use Wave 5 (2015) psychosocial variables to classify whether a
respondent belongs to an operationally defined high
acculturative-stress group in Wave 6 (2016).

This is an educational research project.

It is NOT a clinical diagnostic system and must never describe
predictions as diagnoses or actual intervention decisions.


# Core Research Design

Predictor wave:
MAPS Wave 5, 2015 (middle school grade 2).

Target wave:
MAPS Wave 6, 2016 (middle school grade 3).

Unit of analysis:
one adolescent respondent.

Join waves using the official respondent ID documented in the
MAPS codebook.


# CRITICAL RULE: NEVER GUESS VARIABLE NAMES

Do not infer MAPS variables from similar-looking column names.

Use only variables confirmed by:

1. the official MAPS codebook,
2. the official questionnaire,
3. the raw dataset.

If a variable cannot be verified, stop that variable's processing,
report it in data_quality.md, and continue with verified variables.

Never fabricate reverse-scoring rules, scale ranges, labels,
missing-value codes, or respondent identifiers.


# Raw Data

Raw MAPS files are stored in:

data/raw/

Never modify raw data.

Never commit MAPS raw data to Git.

Derived files may be placed in:

data/interim/
data/processed/

data/demo_format/ contains a 5-row FAKE file used only to smoke-test
the inspector before real data arrives. Its columns are prefixed
DEMO_. It is not MAPS. Never report numbers derived from it.


# Variable Configuration

All mappings between psychological constructs and raw MAPS
columns must live in:

configs/variables.yaml

Analysis code must not contain unexplained hard-coded MAPS
column names.


# Target Construction

Construct the Wave 6 acculturative-stress scale exactly according
to the official MAPS documentation.

If no validated clinical cutoff is provided, define an educational
high-stress group using the 75th percentile of the TRAINING data.

Do not call this a clinical high-risk group.

Use terminology such as:

"operationally defined high-stress group."


# Prediction Models

Implement:

1. DummyClassifier
2. LogisticRegression
3. DecisionTreeClassifier
4. RandomForestClassifier

Do not add neural networks or gradient-boosting libraries unless
explicitly requested.


# Model Sets

Model A:
Wave 5 psychosocial predictors only.

Model B:
The same predictors plus Wave 5 acculturative-stress score.

Compare whether prior acculturative stress adds meaningful
predictive performance.


# Data Leakage Prevention

No Wave 6 variable may be used as a predictor.

Wave 6 variables may only be used to construct the target.

All imputation, encoding, scaling, and model fitting must be
performed inside scikit-learn Pipelines and fit using training
data only.

The final test set must not be used for hyperparameter selection.


# Data Split

Use:

random_state = 42
test_size = 0.20

Stratification: the true high-stress label exists only after the
train-only cutoff (circular dependency), so stratify the split on a
provisional median split of the Wave 6 stress score and document
this choice in code comments.

Feature screening (for example, dropping high-missingness features)
must be computed on the training split only.

Within the training data use stratified 5-fold cross-validation.


# Evaluation

Report:

ROC-AUC
Average Precision / PR-AUC
Recall
Precision
F1
Balanced Accuracy
Confusion Matrix

Do not report accuracy alone.

Always include DummyClassifier as a baseline.


# Explainability

For Logistic Regression:
report standardized coefficients.

For model-agnostic interpretation:
use permutation importance.

SHAP is optional and must not be required for the core pipeline.

Never interpret feature importance as causal effect.


# Reproducibility

The complete pipeline must be runnable from a clean environment.

Provide:

pyproject.toml
README.md

Tests must run using:

pytest -q

The main analysis must run using:

python scripts/build_dataset.py
python scripts/run_models.py


# Required Tests

Implement tests verifying:

- required MAPS columns exist,
- respondent IDs are unique within each wave,
- scale items are within documented ranges,
- Wave 5 and Wave 6 merge correctly,
- no Wave 6 predictor enters X,
- target columns never enter X,
- preprocessing is encapsulated in sklearn Pipeline,
- train/test respondent sets do not overlap incorrectly,
- output metric files are generated.


# Required Outputs

Generate:

reports/data_quality.md
reports/model_metrics.csv
reports/feature_importance.csv

and figures:

class_distribution.png
roc_curve.png
precision_recall_curve.png
confusion_matrix.png
feature_importance.png


# Coding Style

Prefer simple, readable Python suitable for a first-year
psychology student.

Avoid unnecessary abstraction.

Every major function should have a short docstring explaining:

- what it receives,
- what it returns,
- why it exists.

Prefer pandas and scikit-learn.

Do not hide important analysis inside complex frameworks.


# Scientific Integrity

Never fabricate results.

Never insert sample performance values into the final report
unless clearly labeled as examples.

Distinguish:

prediction
association
causation

throughout all generated reports.

Never state that a variable "causes" high acculturative stress
based on predictive importance alone.


# Human Review Gates

Before implementing scale scoring:
request verification of variables.yaml.

Before final modeling:
generate data_quality.md.

Before interpreting final models:
verify class counts, missingness, and baseline metrics.

If any of these are inconsistent, report the problem rather than
silently modifying the analysis.
