import os, sys, pickle, numpy as np, pandas as pd

from imblearn.under_sampling import ClusterCentroids

from pipeline import get_pipeline, CLASSIFIERS


classifier_name = sys.argv[1] if len(sys.argv) > 1 \
    else sys.exit('Please provide a classifier name...')

if classifier_name not in CLASSIFIERS:
    sys.exit(f"Unknown classifier '{classifier_name}'. Choose from: {list(CLASSIFIERS.keys())}")

TRAIN_DATASET_PATH = './02-train-test/train.csv'
ROLES_OF_INTEREST = [ 'button', 'tab', 'combobox',
                      'presentation', 'region', 'link',
                      'menuitem', 'dialog' ]

print('reading dataset')
df = pd.read_csv(TRAIN_DATASET_PATH)

print('dropping na rows')
df = df.dropna(subset=['mutation_role', 'mutation_xpath'])

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

ENCODER_PATH = './03-cv-results/encoder.pkl'
print(f'loading encoder from {ENCODER_PATH}')
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

X_cat = encoder.transform(df[string_columns])
X = np.hstack([df[numeric_columns].to_numpy(), X_cat])
y = labels.values

print('undersampling majority class')
undersampler = ClusterCentroids(sampling_strategy='majority', voting='hard', random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

print('fitting model with grid search CV')
gridsearch_cv = get_pipeline(classifier_name)
gridsearch_cv.fit(X_res, y_res)

print(f'best params: {gridsearch_cv.best_params_}')
print(f'best CV score: {gridsearch_cv.best_score_:.4f}')

OUTPUT_DIR = './04-trained-models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = os.path.join(OUTPUT_DIR, f'{classifier_name}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(gridsearch_cv.best_estimator_, f)
print(f'model saved to {model_path}')


