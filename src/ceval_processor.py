#!/usr/bin/env python3
"""
C-Eval Dataset Processor

This script processes C-Eval dataset parquet files and merges all val data into a single JSON file.
It handles parquet reading errors using PyArrow and provides detailed progress tracking.

C-Eval is a comprehensive Chinese benchmark with 13,948 multiple-choice questions 
covering 52 subjects across four difficulty levels.

Usage:
    python ceval_processor.py --data_dir /path/to/ceval --output output.json
    
Requirements:
    - pyarrow
    - json (built-in)
    - pathlib (built-in)
    - logging (built-in)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: PyArrow is required. Install it with: pip install pyarrow")
    sys.exit(1)


class CEvalProcessor:
    """Processor for C-Eval dataset parquet files"""
    
    def __init__(self, data_dir: str, output_file: str, log_level: str = "INFO"):
        """
        Initialize the processor
        
        Args:
            data_dir: Path to C-Eval dataset directory
            output_file: Output JSON file path
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)
        self.logger = self._setup_logging(log_level)
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('ceval_processor.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def find_subjects(self) -> List[Path]:
        """
        Find all subject directories in the dataset
        
        Returns:
            List of subject directory paths
        """
        subject_dirs = []
        
        # Look for directories that contain parquet files
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if this directory contains val.parquet files
                val_files = list(item.glob("*val*.parquet"))
                if val_files:
                    subject_dirs.append(item)
                    self.logger.debug(f"Found subject directory: {item.name}")
        
        if not subject_dirs:
            # Fallback: look for parquet files directly in subdirectories
            for item in self.data_dir.rglob("*val*.parquet"):
                subject_dir = item.parent
                if subject_dir not in subject_dirs:
                    subject_dirs.append(subject_dir)
                    
        self.logger.info(f"Found {len(subject_dirs)} subject directories")
        return sorted(subject_dirs)
    
    def read_parquet_safely(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Safely read parquet file using PyArrow with error handling
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            List of records or None if reading failed
        """
        try:
            # Try reading with PyArrow
            table = pq.read_table(file_path)
            
            # Convert to list of dictionaries
            records = table.to_pylist()
            
            self.logger.debug(f"Successfully read {len(records)} records from {file_path.name}")
            return records
            
        except Exception as e:
            self.logger.warning(f"PyArrow failed for {file_path.name}: {e}")
            
            # Try alternative reading methods
            try:
                # Try reading with different PyArrow options
                # Note: use_legacy_dataset was removed in newer PyArrow versions
                # Try reading with different parameters instead
                dataset = pq.ParquetDataset(file_path)
                table = dataset.read()
                records = table.to_pylist()
                self.logger.info(f"Successfully read with dataset mode: {file_path.name}")
                return records
                
            except Exception as e2:
                try:
                    # Final fallback: try reading with different engine options
                    with open(file_path, 'rb') as f:
                        table = pq.read_table(f)
                        records = table.to_pylist()
                        self.logger.info(f"Successfully read with file handle: {file_path.name}")
                        return records
                except Exception as e3:
                    self.logger.error(f"All reading methods failed for {file_path.name}: {e}, {e2}, {e3}")
                    return None
    
    def process_subject(self, subject_dir: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single subject directory
        
        Args:
            subject_dir: Path to subject directory
            
        Returns:
            List of records with subject field added, or None if processing failed
        """
        subject_name = subject_dir.name
        self.logger.info(f"Processing subject: {subject_name}")
        
        # Find val parquet files
        val_files = list(subject_dir.glob("*val*.parquet"))
        if not val_files:
            # Try alternative patterns
            val_files = list(subject_dir.glob("val.parquet"))
            if not val_files:
                val_files = list(subject_dir.glob("validation.parquet"))
        
        if not val_files:
            self.logger.warning(f"No val parquet files found in {subject_name}")
            return None
        
        all_records = []
        
        for val_file in val_files:
            self.logger.debug(f"Reading file: {val_file.name}")
            records = self.read_parquet_safely(val_file)
            
            if records is not None:
                # Add subject field to each record
                for record in records:
                    record['subject'] = subject_name
                
                all_records.extend(records)
                self.logger.info(f"Added {len(records)} records from {val_file.name}")
            else:
                self.logger.error(f"Failed to read {val_file.name}")
        
        if all_records:
            self.logger.info(f"Subject {subject_name}: {len(all_records)} total records")
        
        return all_records if all_records else None
    
    def process_all_subjects(self) -> Dict[str, Any]:
        """
        Process all subjects and merge data
        
        Returns:
            Dictionary containing merged data and metadata
        """
        self.logger.info("Starting C-Eval dataset processing...")
        
        subjects = self.find_subjects()
        if not subjects:
            raise ValueError("No subject directories found in the dataset")
        
        all_data = []
        processed_subjects = []
        failed_subjects = []
        
        for i, subject_dir in enumerate(subjects, 1):
            self.logger.info(f"Progress: {i}/{len(subjects)} - Processing {subject_dir.name}")
            
            try:
                subject_data = self.process_subject(subject_dir)
                
                if subject_data:
                    all_data.extend(subject_data)
                    processed_subjects.append(subject_dir.name)
                    self.logger.info(f"✓ {subject_dir.name}: {len(subject_data)} records")
                else:
                    failed_subjects.append(subject_dir.name)
                    self.logger.warning(f"✗ {subject_dir.name}: No data extracted")
                    
            except Exception as e:
                failed_subjects.append(subject_dir.name)
                self.logger.error(f"✗ {subject_dir.name}: Processing failed - {e}")
        
        # Prepare final output
        output_data = {
            "metadata": {
                "total_records": len(all_data),
                "processed_subjects": len(processed_subjects),
                "failed_subjects": len(failed_subjects),
                "subjects_list": processed_subjects,
                "failed_subjects_list": failed_subjects,
                "dataset": "C-Eval",
                "split": "validation"
            },
            "data": all_data
        }
        
        self.logger.info(f"Processing complete: {len(all_data)} total records from {len(processed_subjects)} subjects")
        
        if failed_subjects:
            self.logger.warning(f"Failed to process: {failed_subjects}")
        
        return output_data
    
    def save_json(self, data: Dict[str, Any]) -> None:
        """
        Save data to JSON file
        
        Args:
            data: Data to save
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Data saved to: {self.output_file}")
            self.logger.info(f"File size: {self.output_file.stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON file: {e}")
            raise
    
    def run(self) -> None:
        """Run the complete processing pipeline"""
        try:
            # Process all subjects
            merged_data = self.process_all_subjects()
            
            # Save to JSON
            self.save_json(merged_data)
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Total records: {merged_data['metadata']['total_records']}")
            print(f"Processed subjects: {merged_data['metadata']['processed_subjects']}")
            print(f"Failed subjects: {merged_data['metadata']['failed_subjects']}")
            print(f"Output file: {self.output_file}")
            
            if merged_data['metadata']['failed_subjects_list']:
                print(f"Failed subjects: {', '.join(merged_data['metadata']['failed_subjects_list'])}")
            
            print("="*60)
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process C-Eval dataset parquet files and merge val data to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process dataset in current directory
    python ceval_processor.py --data_dir ./ceval_data --output ceval_val_merged.json
    
    # Process with debug logging
    python ceval_processor.py --data_dir ./ceval_data --output result.json --log_level DEBUG
    
    # Process and save to specific location
    python ceval_processor.py --data_dir /path/to/ceval --output /output/ceval_val.json
        """
    )
    
    parser.add_argument(
        '--data_dir', '-d', 
        required=True,
        help='Path to C-Eval dataset directory containing subject subdirectories'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSON file path for merged data'
    )
    
    parser.add_argument(
        '--log_level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    try:
        processor = CEvalProcessor(args.data_dir, args.output, args.log_level)
        processor.run()
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()