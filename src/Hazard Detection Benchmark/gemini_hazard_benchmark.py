#!/usr/bin/env python3
"""
Hazard Detection Benchmark
Compares Method 1 (manual input) vs Method 2 (AI follow-up) with accuracy evaluation
"""

import os
import json
import base64
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

class GeminiHazardBenchmark:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the benchmark with Gemini API key."""
        genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # SYSTEM PROMPT - @ruben tweak this to your liking
        self.system_prompt = """You are an expert factory safety inspector analyzing hazard photos.

ANALYSIS FOCUS:
- Identify visible safety issues/hazards (electrical, mechanical, chemical, ergonomic, etc.)
- Determine specific location within the facility 
- Note severity and immediate risks
- Suggest corrective actions

CONFIDENCE RULES:
- High confidence (>80%): Clear, obvious hazards with sufficient detail
- Medium confidence (50-80%): Some uncertainty about specifics
- Low confidence (<50%): Unclear image, missing context, or ambiguous hazards

OUTPUT REQUIREMENTS:
- Be specific and safety-focused
- Use technical safety terminology when appropriate
- Consider immediate vs long-term risks
- Prioritize worker safety above all else

RESPONSE FORMAT: Always return valid JSON only."""

    def encode_image(self, image_path: str) -> bytes:
        """Load and prepare image for Gemini API."""
        with open(image_path, "rb") as image_file:
            return image_file.read()

    def method1_manual_input_with_ai(self, image_path: str) -> Dict:
        """Method 1: Manual user input + AI analysis based on that input."""
        print("\n" + "="*60)
        print("METHOD 1: MANUAL INPUT + AI ANALYSIS")
        print("="*60)
        print("Please analyze the image and provide the following information:")
        
        user_issue = input("\n What is the safety issue/hazard? ")
        user_location = input(" Where is it located in the facility? ")
        user_note = input(" Additional notes or details? ")
        
        image_data = self.encode_image(image_path)
        
        ai_prompt = f"""{self.system_prompt}

USER PROVIDED INFORMATION:
- Issue: {user_issue}
- Location: {user_location}  
- Note: {user_note}

TASK: Based on the user's input and your analysis of this image, provide a comprehensive safety assessment.

OUTPUT FORMAT (JSON only):
{{
    "issue": "detailed safety issue based on user input and image analysis",
    "location": "specific location description based on user input and image",
    "note": "comprehensive safety assessment and recommendations",
    "confidence_level": "high|medium|low",
    "capa": "corrective and preventive action recommendation"
}}"""

        try:
            mime_type = "image/png" if image_path.lower().endswith('.png') else "image/jpeg"
            
            response = self.model.generate_content([
                ai_prompt,
                {"mime_type": mime_type, "data": image_data}
            ])
            
            response_text = response.text.strip()
            print(f"\n DEBUG - Method 1 AI Response: {response_text[:200]}...")
            
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            result = json.loads(response_text)
            
            print(f"\n AI Analysis Based on Your Input:")
            print(f"   Issue: {result.get('issue', 'N/A')}")
            print(f"   Location: {result.get('location', 'N/A')}")
            print(f"   Note: {result.get('note', 'N/A')}")
            
            return {
                "method": "manual_input_with_ai",
                "user_input": {
                    "issue": user_issue,
                    "location": user_location,
                    "note": user_note
                },
                "issue": result.get('issue'),
                "location": result.get('location'),
                "note": result.get('note'),
                "confidence_level": result.get('confidence_level'),
                "capa": result.get('capa', ''),
                "source": "ai_analysis_with_user_input"
            }
            
        except json.JSONDecodeError as e:
            print(f" Method 1 JSON parsing error: {str(e)}")
            return {
                "method": "manual_input_with_ai",
                "error": f"JSON parsing failed: {str(e)}",
                "issue": f"Based on user input: {user_issue}",
                "location": user_location,
                "note": user_note
            }
        except Exception as e:
            print(f" Method 1 AI error: {str(e)}")
            return {
                "method": "manual_input_with_ai",
                "error": f"AI analysis failed: {str(e)}",
                "issue": f"Based on user input: {user_issue}",
                "location": user_location,
                "note": user_note
            }

    def method2_ai_followup(self, image_path: str) -> Dict:
        """Method 2: AI analyzes image and asks follow-up question."""
        print("\n" + "="*60)
        print("METHOD 2: AI ANALYSIS WITH FOLLOW-UP")
        print("="*60)
        
        image_data = self.encode_image(image_path)
        
        # First AI call - analyze and ask follow-up question
        followup_prompt = f"""{self.system_prompt}

TASK: Analyze this factory image and generate ONE specific follow-up question that would help you provide a more accurate safety assessment.

OUTPUT FORMAT (JSON only):
{{
    "initial_analysis": "brief description of what you see",
    "confidence_level": "high|medium|low",
    "follow_up_question": "one specific question to improve accuracy",
    "reasoning": "why this question would help"
}}"""

        try:
            # Upload image and get follow-up question
            # Detect image format
            mime_type = "image/png" if image_path.lower().endswith('.png') else "image/jpeg"
            
            response = self.model.generate_content([
                followup_prompt,
                {"mime_type": mime_type, "data": image_data}
            ])
            
            response_text = response.text.strip()
            print(f"\n🔧 DEBUG - Raw AI Response: {response_text[:200]}...")
            
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            followup_result = json.loads(response_text)
            
            print(f"\n AI Initial Analysis: {followup_result.get('initial_analysis', 'N/A')}")
            print(f" Follow-up Question: {followup_result.get('follow_up_question', 'N/A')}")
            
            user_answer = input("\n Your answer: ")
            final_prompt = f"""{self.system_prompt}

CONTEXT: You previously analyzed this image and asked: "{followup_result.get('follow_up_question', '')}"
USER ANSWER: "{user_answer}"

TASK: Now provide your final safety assessment.

OUTPUT FORMAT (JSON only):
{{
    "issue": "specific safety issue/hazard",
    "location": "specific location in facility",
    "note": "additional safety details and severity",
    "confidence_level": "high|medium|low",
    "capa": "corrective and preventive action recommendation"
}}"""

            final_response = self.model.generate_content([
                final_prompt,
                {"mime_type": mime_type, "data": image_data}
            ])
            
            final_text = final_response.text.strip()
            print(f"\n DEBUG - Final AI Response: {final_text[:200]}...")
            
            if "```json" in final_text:
                start = final_text.find("```json") + 7
                end = final_text.find("```", start)
                final_text = final_text[start:end].strip()
            
            final_result = json.loads(final_text)
            
            return {
                "method": "ai_followup",
                "follow_up_question": followup_result.get('follow_up_question'),
                "user_answer": user_answer,
                "issue": final_result.get('issue'),
                "location": final_result.get('location'),
                "note": final_result.get('note'),
                "confidence_level": final_result.get('confidence_level'),
                "capa": final_result.get('capa', ''),
                "source": "ai_analysis_with_followup"
            }
            
        except json.JSONDecodeError as e:
            print(f" JSON parsing error. Raw AI response:")
            print(f"First response: {response.text if 'response' in locals() else 'No response'}")
            print(f"Final response: {final_response.text if 'final_response' in locals() else 'No final response'}")
            return {
                "method": "ai_followup",
                "error": f"JSON parsing failed: {str(e)}",
                "confidence_level": "error",
                "issue": "AI response parsing failed",
                "location": "N/A", 
                "note": "N/A"
            }
        except Exception as e:
            print(f" AI analysis error: {str(e)}")
            return {
                "method": "ai_followup",
                "error": f"AI analysis failed: {str(e)}",
                "confidence_level": "error",
                "issue": "AI analysis failed",
                "location": "N/A",
                "note": "N/A"
            }

    def get_ground_truth(self) -> Dict:
        """Get the actual/correct response from user for accuracy comparison."""
        print("\n" + "="*60)
        print("GROUND TRUTH (Actual Correct Answer)")
        print("="*60)
        print("Based on the image, what is the ACTUAL correct assessment?")
        
        actual_issue = input("\n Actual issue/hazard: ")
        actual_location = input(" Actual location: ")
        actual_note = input(" Actual notes: ")
        
        return {
            "actual_issue": actual_issue,
            "actual_location": actual_location,
            "actual_note": actual_note,
            "source": "ground_truth"
        }

    def evaluate_accuracy(self, method1_result: Dict, method2_result: Dict, ground_truth: Dict) -> Dict:
        """Use AI to evaluate which method was more accurate."""
        print("\n Evaluating accuracy...")
        
        evaluation_prompt = f"""You are an expert evaluator comparing two safety assessment methods against the ground truth.

METHOD 1 RESULT:
Issue: {method1_result.get('issue', 'N/A')}
Location: {method1_result.get('location', 'N/A')}
Note: {method1_result.get('note', 'N/A')}

METHOD 2 RESULT:
Issue: {method2_result.get('issue', 'N/A')}
Location: {method2_result.get('location', 'N/A')}
Note: {method2_result.get('note', 'N/A')}

GROUND TRUTH (CORRECT ANSWER):
Issue: {ground_truth.get('actual_issue', 'N/A')}
Location: {ground_truth.get('actual_location', 'N/A')}
Note: {ground_truth.get('actual_note', 'N/A')}

TASK: Compare each method against the ground truth and determine accuracy percentages.

OUTPUT FORMAT (JSON only):
{{
    "method1_accuracy": 85,
    "method2_accuracy": 92,
    "winner": "method1|method2|tie",
    "method1_analysis": "detailed comparison of method 1 vs ground truth",
    "method2_analysis": "detailed comparison of method 2 vs ground truth",
    "overall_assessment": "which method performed better and why"
}}"""

        try:
            response = self.model.generate_content(evaluation_prompt)
            
            response_text = response.text.strip()
            print(f"\n DEBUG - Evaluation Response: {response_text[:200]}...")
            
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            evaluation = json.loads(response_text)
            return evaluation
        except Exception as e:
            return {
                "error": f"Evaluation failed: {str(e)}",
                "method1_accuracy": 0,
                "method2_accuracy": 0,
                "winner": "error"
            }

    def run_single_benchmark(self, image_path: str, test_number: int) -> Dict:
        """Run a single benchmark test with both methods."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK TEST #{test_number}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        method1_result = self.method1_manual_input_with_ai(image_path)
        
        method2_result = self.method2_ai_followup(image_path)
        
        ground_truth = self.get_ground_truth()
        
        evaluation = self.evaluate_accuracy(method1_result, method2_result, ground_truth)
        
        self._display_results(method1_result, method2_result, evaluation, test_number)
        
        return {
            "test_number": test_number,
            "image_path": image_path,
            "method1_result": method1_result,
            "method2_result": method2_result,
            "ground_truth": ground_truth,
            "evaluation": evaluation
        }

    def _display_results(self, method1: Dict, method2: Dict, evaluation: Dict, test_number: int):
        """Display benchmark results in a clear format."""
        print(f"\n{'='*60}")
        print(f"TEST #{test_number} RESULTS")
        print(f"{'='*60}")
        
        print(f"\n METHOD 1 (Manual Input + AI Analysis):")
        print(f"   {method1.get('issue', 'N/A')}")
        print(f"   Accuracy: {evaluation.get('method1_accuracy', 0)}%")
        
        print(f"\n METHOD 2 (AI Follow-up):")
        print(f"   {method2.get('issue', 'N/A')}")
        print(f"   Accuracy: {evaluation.get('method2_accuracy', 0)}%")
        
        winner = evaluation.get('winner', 'unknown')
        if winner == 'method1':
            print(f"\n WINNER: Method 1 (Manual Input)")
        elif winner == 'method2':
            print(f"\n WINNER: Method 2 (AI Follow-up)")
        else:
            print(f"\n RESULT: Tie or Error")
        
        print(f"\n Assessment: {evaluation.get('overall_assessment', 'N/A')}")

    def run_full_benchmark(self, image_paths: List[str]) -> Dict:
        """Run the complete 5-test benchmark."""
        if len(image_paths) != 5:
            raise ValueError("Please provide exactly 5 image paths for the benchmark")
        
        print(" Starting 5-Test Factory Hazard Detection Benchmark")
        print(" You'll compare Manual Input vs AI Follow-up methods")
        
        results = []
        method1_scores = []
        method2_scores = []
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.run_single_benchmark(image_path, i)
                results.append(result)
                
                # Collect scores
                eval_data = result['evaluation']
                method1_scores.append(eval_data.get('method1_accuracy', 0))
                method2_scores.append(eval_data.get('method2_accuracy', 0))
                
            except Exception as e:
                print(f" Error in test {i}: {str(e)}")
                continue
        
        # stats
        avg_method1 = sum(method1_scores) / len(method1_scores) if method1_scores else 0
        avg_method2 = sum(method2_scores) / len(method2_scores) if method2_scores else 0
        
        self._display_final_summary(avg_method1, avg_method2, method1_scores, method2_scores)
        
        return {
            "benchmark_summary": {
                "total_tests": len(results),
                "method1_average": avg_method1,
                "method2_average": avg_method2,
                "method1_scores": method1_scores,
                "method2_scores": method2_scores,
                "winner": "method1" if avg_method1 > avg_method2 else "method2" if avg_method2 > avg_method1 else "tie"
            },
            "detailed_results": results
        }

    def _display_final_summary(self, avg1: float, avg2: float, scores1: List[float], scores2: List[float]):
        """Display final benchmark summary."""
        print(f"\n{'='*80}")
        print("🏆 FINAL BENCHMARK RESULTS")
        print(f"{'='*80}")
        
        print(f"\n AVERAGE ACCURACY:")
        print(f" Method 1 (Manual Input): {avg1:.1f}%")
        print(f" Method 2 (AI Follow-up): {avg2:.1f}%")
        
        print(f"\n INDIVIDUAL TEST SCORES:")
        for i in range(len(scores1)):
            print(f"   Test {i+1}: Method1={scores1[i]}% | Method2={scores2[i]}%")
        
        if avg1 > avg2:
            print(f"\n🥇 OVERALL WINNER: Method 1 (Manual Input)")
            print(f"   Advantage: {avg1 - avg2:.1f} percentage points")
        elif avg2 > avg1:
            print(f"\n🥇 OVERALL WINNER: Method 2 (AI Follow-up)")
            print(f"   Advantage: {avg2 - avg1:.1f} percentage points")
        else:
            print(f"\n🤝 RESULT: Tie")

def main():
    """Interactive benchmark runner."""
    print("🏭 Gemini Factory Hazard Detection Benchmark")
    print("=" * 50)
    
    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        print(" GEMINI_API_KEY not found in environment variables")
        print("Please set it in .env file or as environment variable")
        return
    
    benchmark = GeminiHazardBenchmark()
    
    print("\nPlease provide 5 factory hazard images for benchmarking:")
    image_paths = []
    
    for i in range(5):
        while True:
            path = input(f"📸 Image {i+1} path: ").strip()
            if os.path.exists(path):
                image_paths.append(path)
                break
            else:
                print(f" File not found: {path}")
    
    try:
        results = benchmark.run_full_benchmark(image_paths)
        
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to: benchmark_results.json")
        
    except Exception as e:
        print(f" Benchmark failed: {str(e)}")

if __name__ == "__main__":
    main()
