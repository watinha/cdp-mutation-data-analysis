import sys, os, pandas as pd


csv_files = os.listdir('./')
df_summary = pd.DataFrame()
df_all = pd.DataFrame()

for filename in csv_files:
    if os.path.getsize(filename) < 5:
        print(f"Skipping empty file: {filename}")
        continue

    if not filename.endswith('.csv'):
        print(f"Skipping non-CSV file: {filename}")
        continue

    print(f"Processing file: {filename}")
    df_url = pd.read_csv(filename)
    (n, m) = df_url.shape
    role_counts = df_url['mutation_role'].value_counts()
    print(role_counts)
    role_counts.name = filename[:-4] # Remove the '.csv' extension

    df_all = pd.concat([df_all, role_counts], axis=1)


roles_sum = df_all.sum(axis=1)
roles_sum.name = 'sum'
roles_count = df_all.count(axis=1)
roles_count.name = 'count'

df_all = df_all.fillna(0).astype(int)
df_all.to_csv('summary-all.csv')
df_summary = pd.concat([df_summary, roles_sum, roles_count], axis=1)
df_summary = df_summary.T
df_summary.to_csv('summary.csv')


sys.exit(0)  # Exit the script successfully
