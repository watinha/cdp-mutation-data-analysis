import os, sys, pickle, json
import numpy as np
import pandas as pd


model_path = sys.argv[1] if len(sys.argv) > 1 \
    else sys.exit('Please provide a path to the trained model pickle file...')

print(f'loading model from {model_path}')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f'model loaded: {model}')

ENCODER_PATH = './03-cv-results/encoder.pkl'
print(f'loading encoder from {ENCODER_PATH}')
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

TEST_URLS_PATH = './02-train-test/test_urls.txt'

print(f'loading test URLs from {TEST_URLS_PATH}')
with open(TEST_URLS_PATH) as f:
    test_urls = json.load(f)

print(f'found {len(test_urls)} test datasets')

DATASETS_DIR = './01-datasets'

RESULTS_DIR = './05-test-results'

for csv_filename in test_urls:
    print(f'\nprocessing {csv_filename}')
    df = pd.read_csv(os.path.join(DATASETS_DIR, csv_filename))

    df = df.dropna(subset=['mutation_xpath'])

    meta = df[['mutation_url', 'mutation_xpath', 'target_xpath',
               'hover_img', 'event_img', 'key_img', 'base_img',
               'target_top', 'target_left', 'target_height', 'target_width',
               'mutation_top', 'mutation_left', 'mutation_height', 'mutation_width']].reset_index(drop=True)

    df['target_role'] = df['target_role'].fillna('none')

    df = df.drop(columns=['target_role', 'mutation_role',
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

    X_cat = encoder.transform(df[string_columns])
    X = np.hstack([df[numeric_columns].to_numpy(), X_cat])
    print(f'  shape: {X.shape}')

    probabilities = model.predict_proba(X)
    prob_df = pd.DataFrame(probabilities, columns=model.classes_)

    output_df = pd.concat([meta, prob_df], axis=1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, csv_filename)
    output_df.to_csv(output_path, index=False)
    print(f'  saved to {output_path}')


