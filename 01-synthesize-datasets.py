import json, os, sys, ast, pandas as pd, gc


DATA_DIR = "./data"
OUTPUT_DIR = "./01-datasets"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



url_folders = os.listdir(DATA_DIR)
for url_folder in url_folders:
    print(f"Processing {url_folder}...")
    files = os.listdir(os.path.join(DATA_DIR, url_folder))

    if not os.path.exists(f'{OUTPUT_DIR}/{url_folder}'):
        os.mkdir(f'{OUTPUT_DIR}/{url_folder}')
    else:
        continue

    dataset = []
    for i_file, file in enumerate(files):
        if file.endswith(".json"):
            print(f"  - Processing {file}...")
            file_path = os.path.join(DATA_DIR, url_folder, file)
            with open(file_path, "r") as f:
                json_str = f.read()
                [target, mutations, hover_img, event_img, key_img] = ast.literal_eval(json_str)

            event_name = file.split('-').pop()[:-5] 

            target_html = f'{OUTPUT_DIR}/{url_folder}/target-{i_file}.html'
            with open(target_html, 'w') as f:
                f.write(target['outerHTML'])
            del target['outerHTML']
            gc.collect()
            target['outerHTML'] = target_html

            for i_mutation, mutation in enumerate(mutations):

                mutation_html = f'{OUTPUT_DIR}/{url_folder}/mutation-{i_file}-{event_name}-{i_mutation}.html'
                with open(mutation_html, 'w') as f:
                    f.write(mutation['outerHTML'])
                del mutation['outerHTML']
                gc.collect()
                mutation['outerHTML'] = mutation_html

                row = { 'event': event_name }
                target_properties = list(target.keys())
                for p in target_properties:
                    row[f"target_{p}"] = target[p]
                
                for p in mutation:
                    row[f"mutation_{p}"] = mutation[p]

                row["hover_img"] = hover_img
                row["event_img"] = event_img
                row["key_img"] = key_img
                row["base_img"] = f'{DATA_DIR}/{url_folder}/screenshot.png'

                dataset.append(row)

            print(f"  - Added {len(mutations)} mutations for {event_name}.")
            print("\n\n")


    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{url_folder}.csv"), index=False)
    print(f"Dataset synthesized with {len(dataset)} entries in {url_folder}.")


sys.exit(0)
