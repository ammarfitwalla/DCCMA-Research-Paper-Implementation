"""
Analyze dataset quality and identify data issues.
Run this to see missing values, data types, and potential issues.
"""

import pandas as pd
import os

def analyze_dataset(tsv_file, split_name):
    """Analyze a dataset split for quality issues."""
    
    print("\n" + "=" * 70)
    print(f"ANALYZING {split_name.upper()} SET")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Column info
    print("\n" + "-" * 70)
    print("COLUMNS:")
    print("-" * 70)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Missing values
    print("\n" + "-" * 70)
    print("MISSING VALUES:")
    print("-" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col:30s}: {missing[col]:5d} ({missing_pct[col]:5.1f}%)")
    
    # Text column analysis
    print("\n" + "-" * 70)
    print("TEXT COLUMN ANALYSIS:")
    print("-" * 70)
    
    # Title
    title_na = df['title'].isna().sum()
    title_empty = (df['title'] == '').sum()
    print(f"title:       NA={title_na:4d}, Empty={title_empty:4d}, Valid={len(df) - title_na - title_empty:4d}")
    
    # Clean title
    clean_na = df['clean_title'].isna().sum()
    clean_empty = (df['clean_title'] == '').sum()
    print(f"clean_title: NA={clean_na:4d}, Empty={clean_empty:4d}, Valid={len(df) - clean_na - clean_empty:4d}")
    
    # Both missing
    both_missing = ((df['title'].isna() | (df['title'] == '')) & 
                   (df['clean_title'].isna() | (df['clean_title'] == ''))).sum()
    print(f"\nSamples with BOTH title and clean_title missing/empty: {both_missing}")
    
    # Image analysis
    print("\n" + "-" * 70)
    print("IMAGE ANALYSIS:")
    print("-" * 70)
    has_image = df['hasImage'].sum()
    no_image = len(df) - has_image
    print(f"Has image:    {has_image:5d} ({has_image/len(df)*100:.1f}%)")
    print(f"No image:     {no_image:5d} ({no_image/len(df)*100:.1f}%)")
    
    image_url_na = df['image_url'].isna().sum()
    image_url_empty = (df['image_url'] == '').sum()
    print(f"\nimage_url NA/empty: {image_url_na + image_url_empty}")
    
    # Label distribution
    print("\n" + "-" * 70)
    print("LABEL DISTRIBUTION:")
    print("-" * 70)
    label_dist = df['2_way_label'].value_counts().sort_index()
    print("2_way_label:")
    for label, count in label_dist.items():
        label_name = "Real" if label == 0 else "Fake"
        print(f"  {label} ({label_name}): {count:5d} ({count/len(df)*100:.1f}%)")
    
    # Data type issues
    print("\n" + "-" * 70)
    print("DATA TYPES:")
    print("-" * 70)
    print(df.dtypes)
    
    # Sample rows with issues
    print("\n" + "-" * 70)
    print("SAMPLE ROWS WITH ISSUES:")
    print("-" * 70)
    
    # Rows with missing text
    missing_text = df[(df['title'].isna() | (df['title'] == '')) & 
                     (df['clean_title'].isna() | (df['clean_title'] == ''))]
    
    if len(missing_text) > 0:
        print(f"\nFound {len(missing_text)} rows with missing text. Sample:")
        print(missing_text[['id', 'title', 'clean_title', 'hasImage', '2_way_label']].head())
    else:
        print("\nNo rows with completely missing text!")
    
    return df

def main():
    """Analyze all dataset splits."""
    
    data_dir = 'dataset/subset'
    
    splits = {
        'train': 'multimodal_train.tsv',
        'validate': 'multimodal_validate.tsv',
        'test': 'multimodal_test_public.tsv'
    }
    
    all_stats = {}
    
    for split_name, filename in splits.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\nWarning: {filepath} not found, skipping...")
            continue
        
        df = analyze_dataset(filepath, split_name)
        all_stats[split_name] = df
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    total_samples = sum(len(df) for df in all_stats.values())
    total_with_images = sum((df['hasImage'] == True).sum() for df in all_stats.values())
    total_missing_text = sum(
        ((df['title'].isna() | (df['title'] == '')) & 
         (df['clean_title'].isna() | (df['clean_title'] == ''))).sum()
        for df in all_stats.values()
    )
    
    print(f"Total samples: {total_samples}")
    print(f"Total with images: {total_with_images}")
    print(f"Total with missing text: {total_missing_text}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
