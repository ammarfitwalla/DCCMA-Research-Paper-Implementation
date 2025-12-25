"""
Create a balanced subset of the Fakeddit dataset for R&D purposes.
This script extracts a smaller, balanced subset from the full dataset:
- Train: 5,000 samples (2,500 fake, 2,500 real)
- Validation: 1,000 samples (500 fake, 500 real)
- Test: 1,000 samples (500 fake, 500 real)
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
DATASET_DIR = "dataset/text_data"
OUTPUT_DIR = "dataset/subset"
SUBSET_SIZES = {
    "train": {"total": 5000, "per_class": 2500},
    "validate": {"total": 1000, "per_class": 500},
    "test": {"total": 1000, "per_class": 500}
}

def create_balanced_subset(input_file, output_file, samples_per_class):
    """
    Create a balanced subset from the input file.
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output TSV file
        samples_per_class: Number of samples per class (fake/real)
    """
    print(f"\nProcessing {input_file}...")
    
    # Read the dataset
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Check label distribution
    label_counts = df['2_way_label'].value_counts()
    print(f"  Label distribution:\n{label_counts}")
    
    # Create balanced subset
    # Label 0 = real, Label 1 = fake (typically)
    subset_frames = []
    
    for label in [0, 1]:
        label_df = df[df['2_way_label'] == label]
        
        # Sample with or without replacement depending on availability
        if len(label_df) >= samples_per_class:
            sampled = label_df.sample(n=samples_per_class, random_state=42)
        else:
            print(f"  Warning: Only {len(label_df)} samples available for label {label}, needed {samples_per_class}")
            sampled = label_df.sample(n=samples_per_class, random_state=42, replace=True)
        
        subset_frames.append(sampled)
    
    # Combine and shuffle
    subset_df = pd.concat(subset_frames, ignore_index=True)
    subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Created subset with {len(subset_df)} samples")
    print(f"  Subset label distribution:\n{subset_df['2_way_label'].value_counts()}")
    
    # Filter to only rows with images (optional - uncomment if you want only samples with images)
    # subset_df = subset_df[subset_df['hasImage'] == True]
    # print(f"  After filtering for images: {len(subset_df)} samples")
    
    # Save subset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subset_df.to_csv(output_file, sep='\t', index=False)
    print(f"  Saved to {output_file}")
    
    return subset_df

def main():
    """Create subsets for train, validation, and test sets."""
    
    print("=" * 60)
    print("Creating Dataset Subsets for DCCMA-Net R&D")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each split
    splits = {
        "train": "multimodal_train.tsv",
        "validate": "multimodal_validate.tsv",
        "test": "multimodal_test_public.tsv"
    }
    
    stats = {}
    
    for split_name, filename in splits.items():
        input_file = os.path.join(DATASET_DIR, filename)
        output_file = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(input_file):
            print(f"\nWarning: {input_file} not found, skipping...")
            continue
        
        samples_per_class = SUBSET_SIZES[split_name]["per_class"]
        subset_df = create_balanced_subset(input_file, output_file, samples_per_class)
        
        stats[split_name] = {
            "total": len(subset_df),
            "with_images": len(subset_df[subset_df['hasImage'] == True]) if 'hasImage' in subset_df.columns else 0,
            "label_0": len(subset_df[subset_df['2_way_label'] == 0]),
            "label_1": len(subset_df[subset_df['2_way_label'] == 1])
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_samples = 0
    total_with_images = 0
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total samples: {split_stats['total']}")
        print(f"  With images: {split_stats['with_images']}")
        print(f"  Label 0 (real): {split_stats['label_0']}")
        print(f"  Label 1 (fake): {split_stats['label_1']}")
        
        total_samples += split_stats['total']
        total_with_images += split_stats['with_images']
    
    print(f"\nTOTAL SUBSET SIZE: {total_samples} samples")
    print(f"TOTAL WITH IMAGES: {total_with_images} samples")
    print(f"\nSubset files saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
