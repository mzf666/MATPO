#!/usr/bin/env python3
"""
Script to combine multiple JSONL files into a single Parquet file.
The step number from the filename is added as a new field.
"""

import json
import pandas as pd
import os
import glob
from pathlib import Path
import argparse

def combine_jsonl_files(input_dir, output_file):
    """
    Combine all JSONL files in a directory into a single Parquet file.
    
    Args:
        input_dir (str): Directory containing JSONL files
        output_file (str): Output Parquet file path
    """
    all_data = []
    
    # Get all JSONL files and sort them by step number
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    jsonl_files.sort(key=lambda x: int(os.path.basename(x).replace('.jsonl', '')))
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        # Extract step number from filename
        filename = os.path.basename(file_path)
        step_number = int(filename.replace('.jsonl', ''))
        
        print(f"Processing {filename} (step {step_number})")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            # Add step number as a new field
                            data['step'] = step_number
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {filename} at line {line_num}: {e}")
                            continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    
    if not all_data:
        print("No data found in any JSONL files")
        return
    
    print(f"Total records: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save as Parquet
    df.to_parquet(output_file, index=False)
    print(f"Successfully saved {len(all_data)} records to {output_file}")
    
    # Print some basic statistics
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if 'step' in df.columns:
        print(f"Step range: {df['step'].min()} - {df['step'].max()}")

def main():
    parser = argparse.ArgumentParser(description='Combine JSONL files into Parquet')
    parser.add_argument('input_dir', help='Directory containing JSONL files')
    parser.add_argument('output_file', help='Output Parquet file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    combine_jsonl_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
