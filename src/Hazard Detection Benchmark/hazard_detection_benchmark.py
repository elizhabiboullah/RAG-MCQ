import os
import json
import base64
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

class HazardDetectionBenchmark:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the benchmark with OpenAI API key."""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API call."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_hazard_image(self, image_path: str) -> Dict:
        """
        Analyze factory hazard image using GPT-4o vision.
        Returns structured output with confidence-based decision making.
        """
        base64_image = self.encode_image(image_path)
        
        system_prompt = """You are an expert factory safety inspector analyzing hazard photos. 

Your task:
1. Identify visible safety issues/hazards
2. Propose bounding boxes for hazards (x, y, width, height as percentages 0-100)
3. Fill three fields: issue, location, note

CONFIDENCE RULES:
- If you can clearly identify specific hazards with high confidence (>80%), provide direct answers
- If confidence is low (<80%) or critical information is missing, generate a clarifying question instead

OUTPUT FORMAT - Always respond with valid JSON only:
{
    "confidence_level": "high|medium|low",
    "mode": "auto_fill|follow_up_question", 
    "issue": "specific safety issue or null if asking question",
    "location": "specific location in facility or null if asking question", 
    "note": "additional safety details or null if asking question",
    "bounding_boxes": [{"x": 0, "y": 0, "width": 0, "height": 0, "label": "hazard description"}],
    "capa": "corrective and preventive action recommendation",
    "follow_up_question": "clarifying question if mode is follow_up_question, otherwise null"
}

Be precise and safety-focused. Only auto-fill if you're highly confident."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this factory image for safety hazards and follow the confidence-based output format."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1  # Low temp, I think this is the most accurate I can get
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up response if it has markdown formatting
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(content)
            
            result['timestamp'] = response.created
            result['model'] = response.model
            result['usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": content,
                "confidence_level": "error",
                "mode": "error"
            }
        except Exception as e:
            return {
                "error": f"API call failed: {str(e)}",
                "confidence_level": "error", 
                "mode": "error"
            }
    
    def run_benchmark(self, image_path: str, output_file: Optional[str] = None) -> Dict:
        """
        Run the complete benchmark analysis on a single image.
        """
        print(f"Analyzing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        result = self.analyze_hazard_image(image_path)
        
        benchmark_result = {
            "benchmark_info": {
                "image_path": image_path,
                "analysis_type": "factory_hazard_detection"
            },
            "analysis_result": result
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(benchmark_result, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        self._print_summary(result)
        
        return benchmark_result
    
    def _print_summary(self, result: Dict):
        """Print a human-readable summary of the analysis."""
        print("\n" + "="*50)
        print("HAZARD DETECTION ANALYSIS SUMMARY")
        print("="*50)
        
        if result.get('error'):
            print(f" ERROR: {result['error']}")
            return
        
        mode = result.get('mode', 'unknown')
        confidence = result.get('confidence_level', 'unknown')
        
        print(f"Mode: {mode.upper()}")
        print(f"Confidence: {confidence.upper()}")
        
        if mode == "auto_fill":
            print(f"\n ISSUE: {result.get('issue', 'N/A')}")
            print(f" LOCATION: {result.get('location', 'N/A')}")
            print(f" NOTE: {result.get('note', 'N/A')}")
            print(f" CAPA: {result.get('capa', 'N/A')}")
            
            boxes = result.get('bounding_boxes', [])
            if boxes:
                print(f"\n🎯 BOUNDING BOXES ({len(boxes)} detected):")
                for i, box in enumerate(boxes, 1):
                    print(f"  {i}. {box.get('label', 'Unknown')} - "
                          f"x:{box.get('x', 0)}%, y:{box.get('y', 0)}%, "
                          f"w:{box.get('width', 0)}%, h:{box.get('height', 0)}%")
        
        elif mode == "follow_up_question":
            question = result.get('follow_up_question', 'No question provided')
            print(f"\n FOLLOW-UP QUESTION:")
            print(f"   {question}")
        
        print("="*50)

def main():
    """Example usage of the benchmark."""
    # Initialize benchmark
    benchmark = HazardDetectionBenchmark()
    
    # Example usage - you'll need to provide an actual image path
    image_path = "sample_hazard_image.jpg"  # Replace with actual image path
    
    if os.path.exists(image_path):
        # Run benchmark
        result = benchmark.run_benchmark(
            image_path=image_path,
            output_file="hazard_analysis_result.json"
        )
        
        # Show performance comparison insight
        mode = result['analysis_result'].get('mode')
        print(f"\n BENCHMARK INSIGHT:")
        if mode == "auto_fill":
            print("Model was confident enough for direct auto-fill")
        elif mode == "follow_up_question":
            print("Model requested clarification before auto-fill")
        else:
            print("Analysis encountered an error")
    else:
        print(f" Please place a factory hazard image at: {image_path}")
        print("Or modify the image_path variable in main() function")

if __name__ == "__main__":
    main()
