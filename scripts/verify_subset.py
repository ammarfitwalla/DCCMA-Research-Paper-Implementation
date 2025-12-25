"""Verify the created subset."""
import pandas as pd
import os

subset_dir = 'dataset/subset'
files = {
    'train': 'multimodal_train.tsv',
    'validate': 'multimodal_validate.tsv',
    'test': 'multimodal_test_public.tsv'
}

print("=" * 60)
print("SUBSET VERIFICATION")
print("=" * 60)

total_samples = 0
total_with_images = 0

for split_name, filename in files.items():
    filepath = os.path.join(subset_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"\nERROR: {filepath} not found!")
        continue
    
    df = pd.read_csv(filepath, sep='\t')
    label_dist = df['2_way_label'].value_counts().to_dict()
    has_images = len(df[df['hasImage'] == True]) if 'hasImage' in df.columns else 0
    
    print(f"\n{split_name.upper()}:")
    print(f"  File: {filename}")
    print(f"  Total samples: {len(df)}")
    print(f"  With images: {has_images}")
    print(f"  Label distribution: {label_dist}")
    
    total_samples += len(df)
    total_with_images += has_images

print(f"\n{'=' * 60}")
print(f"TOTAL: {total_samples} samples")
print(f"TOTAL WITH IMAGES: {total_with_images} samples")
print(f"{'=' * 60}")
