import os, pandas as pd, json


DATA_DIR = './01-datasets'
OUTPUT_DIR = './02-train-test'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test_urls.txt'
test_urls = []

if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)

for csv_filename in os.listdir(DATA_DIR):
  if not csv_filename.endswith('.csv'):
    continue

  try:
    df = pd.read_csv(os.path.join(DATA_DIR, csv_filename))
    counts = df['mutation_role'].value_counts()

    if counts.empty:
      test_urls.append(csv_filename)
    else:
      train_path = os.path.join(OUTPUT_DIR, TRAIN_FILENAME)
      df.to_csv(train_path, mode='a', header=not os.path.exists(train_path), index=False)

  except pd.errors.EmptyDataError:
    print(f"Warning: {csv_filename} is empty and will be skipped.")
    continue
  
with open(os.path.join(OUTPUT_DIR, TEST_FILENAME), 'w') as f:
  f.write(json.dumps(test_urls, indent=2))
