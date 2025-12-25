"""
Data preprocessing and cleaning script.
Run this to create a cleaned version of the dataset.
"""

import pandas as pd
import os
import argparse


def clean_text(text):
    """Clean text data."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).strip()
    
    if len(text) < 3:
        return ""
    
    if text.lower() in ['[deleted]', '[removed]', 'nan', 'none']:
        return ""
    
    return text


def preprocess_dataset(input_file, output_file, 
                       drop_missing_text=False, 
                       drop_missing_image=False,
                       keep_columns=None):
    """
    Preprocess and clean a dataset file.
    
    Args:
        input_file: Path to input TSV
        output_file: Path to output TSV
        drop_missing_text: Drop rows with missing text
        drop_missing_image: Drop rows without images
        keep_columns: List of columns to keep (None = keep all)
    """
    print(f"\nProcessing: {input_file}")
    print("-" * 60)
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    original_size = len(df)
    print(f"Original size: {original_size}")
    
    # Clean text
    print("\nCleaning text columns...")
    df['title'] = df['title'].apply(clean_text)
    df['clean_title'] = df['clean_title'].apply(clean_text)
    
    # Create final_text column
    df['final_text'] = df['clean_title'].where(
        df['clean_title'] != '', 
        df['title']
    )
    
    # Check missing text
    missing_text = df['final_text'] == ''
    print(f"Rows with missing text: {missing_text.sum()}")
    
    if drop_missing_text and missing_text.sum() > 0:
        df = df[~missing_text].reset_index(drop=True)
        print(f"Dropped {missing_text.sum()} rows with missing text")
    
    # Check image availability
    if 'hasImage' in df.columns:
        no_image = df['hasImage'] != True
        print(f"Rows without images: {no_image.sum()}")
        
        if drop_missing_image and no_image.sum() > 0:
            df = df[df['hasImage'] == True].reset_index(drop=True)
            print(f"Dropped {no_image.sum()} rows without images")
    
    # Clean image_url
    if 'image_url' in df.columns:
        df['image_url'] = df['image_url'].fillna('')
    
    # Ensure label is int
    df['2_way_label'] = df['2_way_label'].astype(int)
    
    # Handle numeric columns - fill NaN with median
    numeric_cols = ['num_comments', 'score', 'upvote_ratio']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} NaN values with median: {median_val:.2f}")
    
    # Select columns if specified
    if keep_columns is not None:
        # Ensure essential columns are kept
        essential = ['id', '2_way_label', 'hasImage', 'image_url', 'final_text']
        cols_to_keep = list(set(keep_columns + essential))
        
        # Keep only columns that exist
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        df = df[cols_to_keep]
        print(f"\nKept columns: {cols_to_keep}")
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nFinal size: {len(df)}")
    print(f"Saved to: {output_file}")
    print(f"Removed: {original_size - len(df)} rows")
    
    # Label distribution
    print(f"\nLabel distribution:")
    print(df['2_way_label'].value_counts().to_dict())
    
    return df


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess Fakeddit dataset')
    parser.add_argument('--input-dir', type=str, default='dataset/subset',
                       help='Input directory')
    parser.add_argument('--output-dir', type=str, default='dataset/cleaned',
                       help='Output directory')
    parser.add_argument('--drop-missing-text', action='store_true',
                       help='Drop rows with missing text')
    parser.add_argument('--drop-missing-image', action='store_true',
                       help='Drop rows without images')
    parser.add_argument('--minimal-columns', action='store_true',
                       help='Keep only essential columns')
    
    args = parser.parse_args()
    
    # Define columns to keep if minimal mode
    essential_cols = ['id', 'title', 'clean_title', 'hasImage', 'image_url', '2_way_label']
    metadata_cols = ['num_comments', 'score', 'upvote_ratio', 'created_utc']
    
    keep_cols = essential_cols if args.minimal_columns else None
    
    # Process all splits
    splits = ['multimodal_train.tsv', 'multimodal_validate.tsv', 'multimodal_test_public.tsv']
    
    print("=" * 60)
    print("DATASET PREPROCESSING")
    print("=" * 60)
    
    for split in splits:
        input_file = os.path.join(args.input_dir, split)
        output_file = os.path.join(args.output_dir, split)
        
        if not os.path.exists(input_file):
            print(f"\nSkipping {split} (not found)")
            continue
        
        preprocess_dataset(
            input_file, 
            output_file,
            drop_missing_text=args.drop_missing_text,
            drop_missing_image=args.drop_missing_image,
            keep_columns=keep_cols
        )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
