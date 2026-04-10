import os, pickle, numpy as np, pandas as pd, sys
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
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
df = df.drop(columns=['mutation_role', 'target_role',
                      'mutation_xpath', 'target_xpath',
                      'mutation_label', 'target_label',
                      'mutation_className', 'target_className',
                      'mutation_outerHTML', 'target_outerHTML',
                      'hover_img', 'event_img', 'key_img', 'base_img',
                      'mutation_url', 'target_url',
                      'target_parent_landmark', 'mutation_parent_landmark',
                      'target_mutation_type',
                      'mutation_tagName', 'target_tagName',
                      'target_mutation_attributeName', 'mutation_mutation_attributeName'])


string_columns = ['mutation_mutation_type']
numeric_columns = [c for c in df.columns if c not in string_columns]

RESULTS_DIR = './03-cv-results'
os.makedirs(RESULTS_DIR, exist_ok=True)

encoder_path = os.path.join(RESULTS_DIR, 'encoder.pkl')
if os.path.isfile(encoder_path):
    print(f'loading encoder from {encoder_path}')
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    X_cat = encoder.transform(df[string_columns])
else:
    print('fitting OneHotEncoder on string columns')
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df[string_columns])
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f'encoder saved to {encoder_path}')

cat_feature_names = encoder.get_feature_names_out(string_columns).tolist()

# FEATURE EXTRACTION
X = np.hstack([df[numeric_columns].to_numpy(), X_cat])
feature_names = numeric_columns + cat_feature_names
y = labels.values

cv_strategy = StratifiedShuffleSplit(n_splits=10, random_state=42)

undersampler = ClusterCentroids(sampling_strategy='majority', voting='hard', random_state=42)

print('running CV strategy')
y_true_all, y_pred_all = [], []
selected_features_rows = []
i = 1
for train_idx, test_idx in cv_strategy.split(X, y, groups):
    print('')
    print(f'--- running CV fold... {i}/{cv_strategy.get_n_splits()}')
    print('')
    i += 1

    X_res, y_res = undersampler.fit_resample(X[train_idx], y[train_idx])
    gridsearch_cv = get_pipeline(classifier_name)
    gridsearch_cv.fit(X_res, y_res)
    y_true_all.extend(y[test_idx])
    y_pred_all.extend(gridsearch_cv.predict(X[test_idx]))

    best = gridsearch_cv.best_estimator_
    vt_mask = best.named_steps['variance_threshold'].get_support()
    sel_mask = best.named_steps['selector'].get_support()
    # Compose: start with variance threshold mask, then apply selector mask within surviving features
    full_mask = vt_mask.copy()
    full_mask[vt_mask] = sel_mask
    selected_features_rows.append(full_mask)

RESULTS_PATH = os.path.join(RESULTS_DIR, 'cv_results.xlsx')
RESULTS_ALL_PRED = os.path.join(RESULTS_DIR, 'cv_all_preds.csv')

print(confusion_matrix(y_true_all, y_pred_all))

report = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True)
report_df = pd.DataFrame(report).T

SELECTED_FEATURES_PATH = os.path.join(RESULTS_DIR, 'selected_features.xlsx')

selected_features_df = pd.DataFrame(selected_features_rows, columns=feature_names)

if os.path.isfile(SELECTED_FEATURES_PATH):
    existing_sf = pd.read_excel(SELECTED_FEATURES_PATH, sheet_name=None, index_col=0)
else:
    existing_sf = {}

existing_sf[classifier_name] = selected_features_df

with pd.ExcelWriter(SELECTED_FEATURES_PATH, engine='openpyxl') as writer:
    for sheet_name, sheet_df in existing_sf.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name)

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


