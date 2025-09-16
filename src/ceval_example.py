#!/usr/bin/env python3
"""
Quick example script for C-Eval processor usage

This script demonstrates how to use the C-Eval processor with real or sample data.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from ceval_processor import CEvalProcessor


def example_usage():
    """Example of how to use the C-Eval processor"""
    
    print("C-Eval Processor Usage Example")
    print("=" * 50)
    
    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("   processor = CEvalProcessor(data_dir='./ceval_data', output_file='result.json')")
    print("   processor.run()")
    
    # Example 2: With custom logging
    print("\n2. With Custom Logging:")
    print("   processor = CEvalProcessor(")
    print("       data_dir='./ceval_data',")
    print("       output_file='result.json',") 
    print("       log_level='DEBUG'")
    print("   )")
    print("   processor.run()")
    
    # Example 3: Command line usage
    print("\n3. Command Line Usage:")
    print("   python ceval_processor.py --data_dir ./ceval_data --output result.json")
    print("   python ceval_processor.py -d ./ceval_data -o result.json -l DEBUG")
    
    # Example 4: Error handling
    print("\n4. Error Handling Example:")
    print("""
    try:
        processor = CEvalProcessor(
            data_dir='./ceval_data',
            output_file='./output/ceval_merged.json',
            log_level='INFO'
        )
        processor.run()
        print("Processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Data directory not found: {e}")
    except Exception as e:
        print(f"Processing error: {e}")
    """)
    
    # Example 5: Programmatic usage with validation
    print("\n5. Programmatic Usage with Validation:")
    print("""
    import json
    
    # Process the data
    processor = CEvalProcessor('./ceval_data', './result.json')
    processor.run()
    
    # Validate the output
    with open('./result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total records: {data['metadata']['total_records']}")
    print(f"Subjects processed: {len(data['metadata']['subjects_list'])}")
    
    # Access the actual data
    for record in data['data'][:5]:  # Show first 5 records
        print(f"Subject: {record['subject']}, Question: {record['question'][:50]}...")
    """)
    
    print("\n" + "=" * 50)
    print("For detailed documentation, see: docs/CEVAL_PROCESSOR_README.md")
    print("To run tests: python test_ceval_processor.py")


if __name__ == "__main__":
    example_usage()