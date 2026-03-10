import os
import pandas as pd, sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from pipeline import get_pipeline


classifier_name = sys.argv[1] if len(sys.argv) > 1 \
    else sys.exit('Please provide a classifier name...')

TRAIN_DATASET_PATH = './02-train-test/train.csv'
ROLES_OF_INTEREST = ['button', 'region', 'combobox', 'tab', # appeared in more than 10 websites
                     'presentation', 'menuitem', 'switch',
                     'search', 'group', 'group', 'listbox',
                     'link', 'menu', 'dialog']

df = pd.read_csv(TRAIN_DATASET_PATH)

df = df.dropna(subset=['mutation_role', 'mutation_xpath'])
labels = df['mutation_role']
groups = df['mutation_url']
df['target_role'] = df['target_role'].fillna('none')

df['mutation_role'] = df['mutation_role'].apply(
    lambda r: r if r in ROLES_OF_INTEREST else 'other'
)

df = df.drop(columns=['mutation_role',
                      'mutation_xpath', 'target_xpath',
                      'mutation_label', 'target_label',
                      'mutation_className', 'target_className',
                      'mutation_url', 'target_url' ])


string_columns = ['event', 'target_role', 'target_tagName', 'mutation_tagName',
                  'target_parent_landmark', 'mutation_parent_landmark']

df = pd.get_dummies(df, columns=string_columns)

# FEATURE EXTRACTION
X = df.to_numpy()
y = labels.values

cv_strategy = StratifiedGroupKFold(n_splits=3)

y_true_all, y_pred_all = [], []
for train_idx, test_idx in cv_strategy.split(X, y, groups):
    gridsearch_cv = get_pipeline(classifier_name)
    gridsearch_cv.fit(X[train_idx], y[train_idx])
    y_true_all.extend(y[test_idx])
    y_pred_all.extend(gridsearch_cv.predict(X[test_idx]))

RESULTS_DIR = './03-cv-results'
RESULTS_PATH = os.path.join(RESULTS_DIR, 'cv_results.xlsx')

print(confusion_matrix(y_true_all, y_pred_all))

report = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True)
report_df = pd.DataFrame(report).T

os.makedirs(RESULTS_DIR, exist_ok=True)

if os.path.isfile(RESULTS_PATH):
    existing = pd.read_excel(RESULTS_PATH, sheet_name=None, index_col=0)
else:
    existing = {}

if classifier_name in existing:
    existing[classifier_name] = pd.concat([existing[classifier_name], report_df])
else:
    existing[classifier_name] = report_df

with pd.ExcelWriter(RESULTS_PATH, engine='openpyxl') as writer:
    for sheet_name, sheet_df in existing.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name)

print(classification_report(y_true_all, y_pred_all, digits=4))


