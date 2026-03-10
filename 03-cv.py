import pandas as pd, sys


from pipeline import get_pipeline
from sklearn.model_selection import GroupShuffleSplit, cross_validate


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

gridsearch_cv = get_pipeline(classifier_name)

cv_strategy = GroupShuffleSplit(n_splits=5, random_state=42)
cv_results = cross_validate(
    gridsearch_cv, X, y,
    groups=groups,
    cv=cv_strategy,
    scoring={
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
    },
    return_estimator=True,
)

print(cv_results)


