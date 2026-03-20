import os
import pandas as pd, sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import ClusterCentroids

from pipeline import get_pipeline


classifier_name = sys.argv[1] if len(sys.argv) > 1 \
    else sys.exit('Please provide a classifier name...')

TRAIN_DATASET_PATH = './02-train-test/train.csv'
ROLES_OF_INTEREST = [ 'button', 'tab', 'combobox', # appeared in more than 10 websites
                      'presentation', 'region', 'link',
                      'menuitem', 'dialog' ]

print('reading dataset')
df = pd.read_csv(TRAIN_DATASET_PATH)

print('droping na rows')
df = df.dropna(subset=['mutation_role', 'mutation_xpath'])
groups = df['mutation_url']

print('filling na in target_role')
df['target_role'] = df['target_role'].fillna('none')

print('looking only into roles with expressive numbers...')
df['mutation_role'] = df['mutation_role'].apply(
    lambda r: r if r in ROLES_OF_INTEREST else 'other'
)
labels = df['mutation_role']

print('dropping columns which will not be used')
df = df.drop(columns=['mutation_role',
                      'mutation_xpath', 'target_xpath',
                      'mutation_label', 'target_label',
                      'mutation_className', 'target_className',
                      'mutation_outerHTML', 'target_outerHTML',
                      'hover_img', 'event_img', 'key_img', 'base_img',
                      'mutation_url', 'target_url',
                      'target_mutation_type', 'mutation_tagName',
                      'target_mutation_attributeName', 'mutation_mutation_attributeName'])


string_columns = ['event', 'target_role',
                  'target_tagName', #'mutation_tagName',
                  'target_parent_landmark', 'mutation_parent_landmark',
                  'mutation_mutation_type']

print('transforming string columns using get_dummies')
df = pd.get_dummies(df, columns=string_columns)

# FEATURE EXTRACTION
X = df.to_numpy()
y = labels.values

cv_strategy = StratifiedShuffleSplit(n_splits=10)

undersampler = ClusterCentroids(sampling_strategy='majority', voting='hard')

print('running CV strategy')
y_true_all, y_pred_all = [], []
for train_idx, test_idx in cv_strategy.split(X, y, groups):
    X_res, y_res = undersampler.fit_resample(X[train_idx], y[train_idx])
    gridsearch_cv = get_pipeline(classifier_name)
    gridsearch_cv.fit(X_res, y_res)
    y_true_all.extend(y[test_idx])
    y_pred_all.extend(gridsearch_cv.predict(X[test_idx]))

RESULTS_DIR = './03-cv-results'
RESULTS_PATH = os.path.join(RESULTS_DIR, 'cv_results.xlsx')
RESULTS_ALL_PRED = os.path.join(RESULTS_DIR, 'cv_all_preds.csv')

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

all_preds_df = pd.DataFrame()
all_preds_df['y_true'] = y_true_all
if os.path.isfile(RESULTS_ALL_PRED):
    all_preds_df = pd.read_csv(RESULTS_ALL_PRED, index_col=0)
all_preds_df[classifier_name] = y_pred_all
all_preds_df.to_csv(RESULTS_ALL_PRED)


print(classification_report(y_true_all, y_pred_all, digits=3))


