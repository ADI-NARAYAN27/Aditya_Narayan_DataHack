import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

RANDOM_SEED = 6

# Load training datasets
training_features = '/content/training_set_features.csv'
training_labels = '/content/training_set_labels.csv'

# Load training features and labels
features_df = pd.read_csv(training_features, index_col='respondent_id')
labels_df = pd.read_csv(training_labels, index_col='respondent_id')

# Preprocessing pipeline
numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols)
    ])

# Define the multi-output classifier model
model = MultiOutputClassifier(
    estimator=LogisticRegression(penalty='l2', C=1, max_iter=1000, random_state=RANDOM_SEED)
)

# Full pipeline combining preprocessing and model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split data for training and evaluation
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

# Predict probabilities on the evaluation set
preds = full_pipeline.predict_proba(X_eval)

# Format predictions into DataFrame
y_preds = pd.DataFrame(
    {
        'xyz_vaccine': preds[0][:, 1],
        'seasonal_vaccine': preds[1][:, 1]
    },
    index=y_eval.index
)

# Evaluate the model using ROC AUC
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

plot_roc(y_eval['xyz_vaccine'], y_preds['xyz_vaccine'], 'xyz_vaccine', ax=ax[0])
plot_roc(y_eval['seasonal_vaccine'], y_preds['seasonal_vaccine'], 'seasonal_vaccine', ax=ax[1])
fig.tight_layout()
plt.show()

print(f"Overall ROC AUC score: {roc_auc_score(y_eval, y_preds, average='macro'):.4f}")

# Train the model on the full dataset
full_pipeline.fit(features_df, labels_df)

# Load the test set features
test_features_df = pd.read_csv('/content/test_set_features.csv', index_col='respondent_id')

# Predict probabilities on the test set
test_probas = full_pipeline.predict_proba(test_features_df)

# Prepare the submission DataFrame
submission_df = pd.read_csv('/content/submission_format.csv', index_col='respondent_id')
submission_df['xyz_vaccine'] = test_probas[0][:, 1]
submission_df['seasonal_vaccine'] = test_probas[1][:, 1]

# Reorder columns as specified
submission_df = submission_df[['xyz_vaccine', 'seasonal_vaccine']]

# Save the submission DataFrame to CSV without 'h1n1_vaccine' column
submission_df.to_csv('/content/submission_final.csv', index=True, index_label='respondent_id')

# Display the first few lines of the final submission file
!head /content/submission_final.csv
