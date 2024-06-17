import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Set a random seed for reproducibility
RANDOM_SEED = 6

# Load the dataset (assuming features_df and labels_df are already loaded)
# features_df = pd.read_csv('features.csv')  # Example loading step
# labels_df = pd.read_csv('labels.csv')      # Example loading step

# Select numeric columns
numeric_cols = features_df.select_dtypes(include=[np.number]).columns

# Define preprocessing steps
numeric_preprocessing_steps = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('simple_imputer', SimpleImputer(strategy='median'))
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessing_steps, numeric_cols)
    ],
    remainder="drop"
)

# Define the estimators
estimators = MultiOutputClassifier(
    estimator=LogisticRegression(penalty="l2", C=1, random_state=RANDOM_SEED)
)

# Combine into a full pipeline
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("estimators", estimators),
])

# Split the dataset
X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

# Train the model
full_pipeline.fit(X_train, y_train)

# Predict on evaluation set
preds = full_pipeline.predict_proba(X_eval)

# Extract probabilities for the second column (class 1)
y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[0][:, 1],
        "seasonal_vaccine": preds[1][:, 1],
    },
    index=y_eval.index
)

# Plot ROC curves
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
plot_roc(y_eval['h1n1_vaccine'], y_preds['h1n1_vaccine'], 'h1n1_vaccine', ax=ax[0])
plot_roc(y_eval['seasonal_vaccine'], y_preds['seasonal_vaccine'], 'seasonal_vaccine', ax=ax[1])
fig.tight_layout()
plt.show()

# Calculate the overall AUC
overall_auc = roc_auc_score(y_eval, y_preds)
print(f"Overall AUC: {overall_auc:.4f}")

# Retrain model on full dataset
full_pipeline.fit(features_df, labels_df)

# Make predictions on the test set
test_features_df = pd.read_csv('path_to_test_set_features.csv', index_col="respondent_id")
test_probas = full_pipeline.predict_proba(test_features_df)

# Prepare submission
submission_df = pd.read_csv('path_to_submission_format.csv', index_col="respondent_id")
submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
submission_df["seasonal_vaccine"] = test_probas[1][:, 1]

# Save submission
submission_df.to_csv('my_submission.csv', index=True)
