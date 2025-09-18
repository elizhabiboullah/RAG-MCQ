#!/usr/bin/env python3
"""
for testing individual images
Usage: python quick_benchmark.py <image_path>
"""

import sys
import os
from gemini_hazard_benchmark import GeminiHazardBenchmark

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_benchmark.py <image_path>")
        print("Example: python quick_benchmark.py factory_hazard.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print(" GEMINI_API_KEY not found in environment variables")
        print("Please set it in .env file or as environment variable")
        sys.exit(1)
    
    print(f" Running single test benchmark on: {image_path}")
    
    try:
        benchmark = GeminiHazardBenchmark()
        result = benchmark.run_single_benchmark(image_path, 1)
        
        with open("single_test_result.json", "w") as f:
            import json
            json.dump(result, f, indent=2)
        
        print(f" Result saved to: single_test_result.json")
        
    except Exception as e:
        print(f" Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
