#!/usr/bin/env python3
"""
Simple test script to quickly validate the hazard detection benchmark.
Run this with: python simple_test.py <image_path>
"""

import sys
import os
from hazard_detection_benchmark import HazardDetectionBenchmark

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_test.py <image_path>")
        print("Example: python simple_test.py factory_hazard.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print(" OPENAI_API_KEY not found in environment variables")
        print("Please set it in .env file or as environment variable")
        sys.exit(1)
    
    print(f" Testing hazard detection on: {image_path}")
    
    try:
        benchmark = HazardDetectionBenchmark()
        result = benchmark.run_benchmark(image_path)
        
        mode = result['analysis_result'].get('mode')
        confidence = result['analysis_result'].get('confidence_level')
        
        print(f" BENCHMARK RESULT:")
        print(f" Decision: {mode}")
        print(f" Confidence: {confidence}")
        
        if mode == "auto_fill":
            print(" Model auto-filled all fields")
        elif mode == "follow_up_question":
            print(" Model requested clarification")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
