import json, os, sys, ast, pandas as pd


DATA_DIR = "./data"
OUTPUT_DIR = "./01-datasets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



url_folders = os.listdir(DATA_DIR)
for url_folder in url_folders:
    print(f"Processing {url_folder}...")
    files = os.listdir(os.path.join(DATA_DIR, url_folder))

    dataset = []
    for file in files:
        if file.endswith(".json"):
            print(f"  - Processing {file}...")
            file_path = os.path.join(DATA_DIR, url_folder, file)
            with open(file_path, "r") as f:
                json_str = f.read()
                [target, mutations] = ast.literal_eval(json_str)

            event_name = file.split('-').pop()[:-5] 
            for mutation in mutations:
                row = { 'event': event_name }
                target_properties = list(target.keys())
                for p in target_properties:
                    row[f"target_{p}"] = target[p]
                
                for p in mutation:
                    row[f"mutation_{p}"] = mutation[p]

                dataset.append(row)

            print(f"  - Added {len(mutations)} mutations for {event_name}.")
            print("\n\n")


    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{url_folder}.csv"), index=False)
    print(f"Dataset synthesized with {len(dataset)} entries in {url_folder}.")


sys.exit(0)
