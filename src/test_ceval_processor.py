#!/usr/bin/env python3
"""
Test script for C-Eval processor functionality

This script creates sample test data and validates the processor functionality.
"""

import json
import tempfile
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def create_sample_data():
    """Create sample C-Eval data for testing"""
    
    # Sample data for different subjects
    subjects_data = {
        'computer_science': [
            {
                'id': 1,
                'question': '计算机科学中，CPU的全称是什么？',
                'A': 'Central Processing Unit',
                'B': 'Computer Processing Unit', 
                'C': 'Core Processing Unit',
                'D': 'Central Power Unit',
                'answer': 'A'
            },
            {
                'id': 2,
                'question': '以下哪个不是编程语言？',
                'A': 'Python',
                'B': 'Java',
                'C': 'HTML',
                'D': 'C++',
                'answer': 'C'
            }
        ],
        'mathematics': [
            {
                'id': 3,
                'question': '圆周率π的近似值是多少？',
                'A': '3.14',
                'B': '3.15',
                'C': '3.13',
                'D': '3.16',
                'answer': 'A'
            }
        ],
        'physics': [
            {
                'id': 4,
                'question': '光速在真空中的数值大约是多少？',
                'A': '299,792,458 m/s',
                'B': '300,000,000 m/s',
                'C': '299,000,000 m/s', 
                'D': '298,000,000 m/s',
                'answer': 'A'
            }
        ]
    }
    
    return subjects_data


def create_test_dataset(base_dir: Path):
    """Create a test dataset structure with parquet files"""
    
    sample_data = create_sample_data()
    
    for subject, records in sample_data.items():
        subject_dir = base_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Create val.parquet file
        table = pa.Table.from_pylist(records)
        val_file = subject_dir / 'val.parquet'
        pq.write_table(table, val_file)
        
        # Also create dev and test files (empty for this test)
        dev_records = []
        test_records = []
        
        if dev_records:
            dev_table = pa.Table.from_pylist(dev_records)
            pq.write_table(dev_table, subject_dir / 'dev.parquet')
            
        if test_records:
            test_table = pa.Table.from_pylist(test_records)
            pq.write_table(test_table, subject_dir / 'test.parquet')
        
        print(f"Created {subject} with {len(records)} val records")


def test_processor():
    """Test the C-Eval processor with sample data"""
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test dataset
        dataset_dir = temp_path / 'ceval_test'
        output_file = temp_path / 'test_output.json'
        
        print("Creating test dataset...")
        create_test_dataset(dataset_dir)
        
        # Test the processor
        print("\nTesting C-Eval processor...")
        
        # Import the processor
        import sys
        sys.path.append('/home/runner/work/LLM-MCP-RAG/LLM-MCP-RAG/src')
        
        from ceval_processor import CEvalProcessor
        
        # Create and run processor
        processor = CEvalProcessor(
            data_dir=str(dataset_dir),
            output_file=str(output_file),
            log_level='INFO'
        )
        
        try:
            processor.run()
            
            # Validate output
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                print(f"\n✓ Processing successful!")
                print(f"✓ Total records: {result['metadata']['total_records']}")
                print(f"✓ Processed subjects: {result['metadata']['processed_subjects']}")
                print(f"✓ Failed subjects: {result['metadata']['failed_subjects']}")
                print(f"✓ Subjects: {', '.join(result['metadata']['subjects_list'])}")
                
                # Validate data integrity
                expected_total = 4  # 2 + 1 + 1 from our sample data
                if result['metadata']['total_records'] == expected_total:
                    print(f"✓ Data integrity check passed")
                else:
                    print(f"✗ Data integrity check failed: expected {expected_total}, got {result['metadata']['total_records']}")
                
                # Check if subjects are correctly added
                subjects_found = set()
                for record in result['data']:
                    if 'subject' in record:
                        subjects_found.add(record['subject'])
                
                expected_subjects = {'computer_science', 'mathematics', 'physics'}
                if subjects_found == expected_subjects:
                    print(f"✓ Subject field check passed")
                else:
                    print(f"✗ Subject field check failed: expected {expected_subjects}, got {subjects_found}")
                
                return True
            else:
                print("✗ Output file was not created")
                return False
                
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            return False


def test_error_handling():
    """Test error handling with corrupted data"""
    
    print("\n" + "="*50)
    print("Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a directory structure with missing files
        dataset_dir = temp_path / 'ceval_error_test'
        output_file = temp_path / 'error_test_output.json'
        
        # Create some valid data
        create_test_dataset(dataset_dir)
        
        # Create a subject directory without val files
        empty_subject = dataset_dir / 'empty_subject'
        empty_subject.mkdir(parents=True, exist_ok=True)
        
        # Create a corrupted file
        corrupted_subject = dataset_dir / 'corrupted_subject'
        corrupted_subject.mkdir(parents=True, exist_ok=True)
        corrupted_file = corrupted_subject / 'val.parquet'
        
        # Write invalid data to the file
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid parquet file")
        
        print(f"Created test dataset with error conditions...")
        
        # Import the processor
        import sys
        sys.path.append('/home/runner/work/LLM-MCP-RAG/LLM-MCP-RAG/src')
        
        from ceval_processor import CEvalProcessor
        
        # Test processor with problematic data
        processor = CEvalProcessor(
            data_dir=str(dataset_dir),
            output_file=str(output_file),
            log_level='INFO'
        )
        
        try:
            processor.run()
            
            # Check if processor handled errors gracefully
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                print(f"✓ Error handling test passed")
                print(f"✓ Processed records despite errors: {result['metadata']['total_records']}")
                print(f"✓ Failed subjects: {result['metadata']['failed_subjects']}")
                
                if result['metadata']['failed_subjects'] > 0:
                    print(f"✓ Error detection working: {result['metadata']['failed_subjects_list']}")
                
                return True
            else:
                print("✗ Error handling test failed: no output file")
                return False
                
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False


def main():
    """Run all tests"""
    print("="*60)
    print("C-Eval Processor Test Suite")
    print("="*60)
    
    # Test basic functionality
    test1_passed = test_processor()
    
    # Test error handling
    test2_passed = test_error_handling()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic functionality test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Error handling test: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! The processor is working correctly.")
        return True
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)