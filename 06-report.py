import os
import pandas as pd
from PIL import Image, ImageDraw


RESULTS_DIR = './05-test-results'
REPORT_DIR = './06-report-results'

META_COLUMNS = ['mutation_url', 'mutation_xpath', 'target_xpath',
                'hover_img', 'event_img', 'key_img', 'base_img',
                'target_top', 'target_left', 'target_height', 'target_width',
                'mutation_top', 'mutation_left', 'mutation_height', 'mutation_width']

PROBABILITY_THRESHOLD = 0.3

IMG_COLUMNS = ['event_img', 'key_img', 'hover_img', 'base_img']


def draw_annotation(img_path, top, left, height, width, output_path):
    img = Image.open(img_path).convert('RGBA')
    img_w, img_h = img.size

    if height == 0 or width == 0 or top > img_h or top < 0 or left > img_w or left < 0: return

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Get image dimensions
    l = int(left)
    t = int(top)
    r = int(left + width)
    b = int(top + height)
    # Draw rectangle annotation
    draw.rectangle(
        [l, t, r, b],
        outline=(255, 0, 0, 255),
        width=3,
    )
    annotated = Image.alpha_composite(img, overlay).convert('RGB')
    annotated.save(output_path)

    try:
        img_w, img_h = img.size
        l = max(0, int(left))
        t = max(0, int(top))
        r = min(img_w, int(left + width))
        b = min(img_h, int(top + height))
        if r > l and b > t:
            # Reopen original (RGB) for crop to avoid alpha issues
            orig = Image.open(img_path).convert('RGB')
            cropped = orig.crop((l, t, r, b))
            base, ext = os.path.splitext(output_path)
            target_out = f"{base}.target.png"
            cropped.save(target_out)
    except Exception:
        # If cropping fails, skip without raising to avoid stopping the report generation
        print(f' - {output_path}: did not cut')
        pass


os.makedirs(REPORT_DIR, exist_ok=True)

for csv_filename in os.listdir(RESULTS_DIR):
    if not csv_filename.endswith('.csv'):
        continue

    df = pd.read_csv(os.path.join(RESULTS_DIR, csv_filename))

    role_columns = [c for c in df.columns if c not in META_COLUMNS]

    # Keep rows where any role has probability > threshold
    mask = (df[role_columns] > PROBABILITY_THRESHOLD).any(axis=1)
    filtered_df = df[mask].copy()

    # Add a predicted_role column indicating the role with highest probability
    filtered_df['predicted_role'] = filtered_df[role_columns].idxmax(axis=1)

    # Drop rows where the predicted role is "other"
    filtered_df = filtered_df[filtered_df['predicted_role'] != 'other'].copy()

    output_dir = os.path.join(REPORT_DIR, os.path.splitext(csv_filename)[0])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predicted_roles.csv')
    filtered_df.to_csv(output_path, index=False)
    print(f'{csv_filename}: {len(filtered_df)}/{len(df)} rows kept ("other" roles excluded)')

    for i, row in enumerate(filtered_df.itertuples(index=False)):
        top = row.mutation_top
        left = row.mutation_left
        height = row.mutation_height
        width = row.mutation_width
        role = row.predicted_role

        for img_col in IMG_COLUMNS:
            img_path = getattr(row, img_col)
            if pd.isna(img_path) or not os.path.isfile(img_path):
                continue
            out_img_path = os.path.join(output_dir, f'{i}_{role}_{img_col}.png')
            draw_annotation(img_path, top, left, height, width, out_img_path)




