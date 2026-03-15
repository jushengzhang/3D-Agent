import os
import requests
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import re
import base64
import json
import time
class VLMPreliminaryAnnotation(nn.Module):
    def __init__(self, 
                 api_key: str = " ",
                 num_candidates: int = 5,
                 temperature: float = 0.7,
                 logprobs_field_name: str = "logprobs",
                 top_logprobs_field_name: str = "top_logprobs",
                 output_file: str = "annotation_results.json"
                 ):
        super().__init__()
        self.api_key = api_key
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.logprobs_field_name = logprobs_field_name
        self.top_logprobs_field_name = top_logprobs_field_name
        self.output_file = output_file

    def _parse_view_name(self, path: str) -> str:
        """Extract the standard view name from the path"""
        base = os.path.splitext(os.path.basename(path))[0]
        
        # Standard view mapping table (compliant with engineering drawing specifications)
        view_mapping = {
            "0": "Front", "1": "Back", "2": "Left",
            "3": "Right", "4": "Up", "5": "Down"
        }
        
        # Prioritize matching numeric identifiers
        if match := re.search(r"view_(\d)", base, re.IGNORECASE):
            return view_mapping.get(match.group(1), "Unknown")
            
        # Match standard naming (case insensitive)
        name_mapping = {
            r"front": "Front",
            r"back|rear": "Back",
            r"left": "Left",
            r"right": "Right",
            r"top|up": "Up",
            r"bottom|down": "Down"
        }
        
        for pattern, name in name_mapping.items():
            if re.search(pattern, base, re.IGNORECASE):
                return name
                
        # Default handling (capitalize the first letter)
        return base.replace("_", " ").title()

    def _build_multi_turn_prompt(self, image_path: str, base64_image: str) -> List[Dict]:
        """构建简单的多轮提示"""
        view_name = self._parse_view_name(image_path)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Let's analyze this 3D model step by step from the {view_name} perspective:\n"
                                + ("1. What is this object?(If there is a specific name, please describe it with the specific name)\n" if view_name in ["Front", "Back"] else "Don't define what this thing is, and don't define it in the following points.") +
                                "2.The material of the object (e.g. wood, metal, plastic, etc.).\n"
                                "3.The color, shape, and appearance details of the object (e.g. whether it has carvings, smooth surface, etc.).\n"
                                "4.The function and possible use of the object (e.g. use as a desk, dining table, etc.).\n"
                                "5.The scene where the object is located (e.g. placed in a living room, outdoor garden, etc.).\n"
                                "Summarize these five points into a complete paragraph for me"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            },
            # {
            #     "role": "assistant",
            #     "content": "Let me provide a brief analysis:\n"
            #               "[The model will fill in the analysis process, with brief answers for each point, and finally provide a complete paragraph]"
            # }
        ]

    def forward(self, image_paths: List[str]) -> Dict[int, List[Tuple[str, float]]]:
        """Process input paths from 6 perspectives and return candidate responses and confidence scores"""
        results = {}
        output_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "qwen/qwen2.5-vl-72b-instruct"
            },
            "results": {}
        }
        
        for idx, path in enumerate(image_paths):
            relative_path = os.path.relpath(path)  # Convert to relative path
            if not os.path.exists(path):
                print(f"Warning: Image path does not exist: {path}")
                results[idx] = [("Invalid path", 0.0)]
                continue

            print(f"Processing image: {path}")
            
            def image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/png;base64,{encoded_string}"  # Modify based on your image type

            base64_image = image_to_base64(path)  # Generate base64 first

            all_candidates = []  # Store all candidate results
            for _ in range(5): # Loop 5 times
                # Prepare request data
                # Modify model name and parameter settings
                json_data = {
                    "model": "qwen/qwen2.5-vl-72b-instruct",
                    "messages": self._build_multi_turn_prompt(path, base64_image),
                    "temperature": self.temperature,
                    "max_tokens": 1024,
                    "logprobs": True,  # Request log probabilities
                    "top_logprobs": 3,  # Request top 3 most likely tokens and their probabilities
                    "return_logprobs": True  # Ensure log probabilities are returned
                }

                try:
                    response = requests.post(
                        self.api_url,
                        headers=self.headers,
                        json=json_data,
                        timeout=30
                    )
                    # print(f"API response status code: {response.status_code}")  # Debug log

                    # Detailed recording of raw response (note to hide sensitive information)
                    raw_response = response.text
                    # print(f"Complete response content: {raw_response}")  # Print complete response

                    response.raise_for_status()
                    response_json = response.json()

                    # Print response structure
                    # print(f"Response structure: {response_json.keys()}")

                    # Check if choices exists
                    if 'choices' in response_json:
                        # Initialize candidate answer list before iterating through choices
                        candidates = []
                        for i, choice in enumerate(response_json['choices']):
                            if choice is None:
                                print(f"choice {i} is None")
                                candidates.append(("Empty choice", 0.0))
                                continue

                            message = choice.get('message', {})
                            content = message.get('content', "No content")

                            logprobs = choice.get('logprobs', {})
                            top_logprobs = choice.get('top_logprobs', {})

                            if isinstance(logprobs, dict) and 'content' in logprobs:
                                # Extract logprobs for all tokens from the content list
                                token_logprobs = []
                                for entry in logprobs['content']:
                                    if isinstance(entry, dict) and 'logprob' in entry:
                                        token_logprobs.append(entry['logprob'])

                                if token_logprobs:
                                    avg_logprob = sum(token_logprobs) / len(token_logprobs)
                                else:
                                    avg_logprob = 0.0
                                    print("Warning: logprobs.content is empty or has no logprob values")
                            elif isinstance(top_logprobs, list) and top_logprobs:
                                # If top_logprobs is at the choice level (unlikely here), as a fallback
                                flat_top_logprobs = [logprob for sublist in top_logprobs for logprob in sublist]
                                if flat_top_logprobs:
                                    avg_logprob = sum(flat_top_logprobs) / len(flat_top_logprobs)
                                else:
                                    avg_logprob = 0.0
                            else:
                                avg_logprob = 0.0
                                print("Warning: Response is missing logprobs and top_logprobs fields")

                            # Add candidate answer and calculated score to the list
                            candidates.append((content, abs(avg_logprob)))
                        all_candidates.extend(candidates) # Add results to all_candidates

                    else:
                        print(f"Error: Response lacks choices field, complete response: {response_json}")
                        all_candidates.append(("API Error: No choices", 0.0)) # Add error message to all_candidates
                        continue


                except requests.exceptions.RequestException as e:
                    print(f"API request failed: {str(e)}\nRequest details: {json_data}")  # Log request parameters
                    all_candidates.append(("Request Exception", 0.0)) # Add error message to all_candidates
                except json.JSONDecodeError:
                    print(f"Response parsing failed, raw content: {raw_response}")
                    all_candidates.append(("Invalid JSON", 0.0)) # Add error message to all_candidates
                except Exception as e:
                    print(f"Uncaught exception: {type(e).__name__} - {str(e)}")
                    all_candidates.append(("Unexpected Error", 0.0)) # Add error message to all_candidates
            
                time.sleep(1) # Wait 1 second
            
                # Add results to output_data
            output_data["results"][relative_path] = {
                "view_index": idx,
                "view_name": self._parse_view_name(path),
                "candidates": [
                    {
                        "text": text,
                        "confidence_score": float(score),  # Ensure score can be JSON serialized
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    } for text, score in all_candidates
                ]
            }

            results[idx] = all_candidates # Assign all_candidates to results[idx]

            # Save results to JSON file
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Results saved to: {self.output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            
        return results

    def predict_step(self, batch: List[str], batch_idx: int) -> Dict:
        """PyTorch Lightning prediction step"""
        return self(batch)

# Example usage (using scheme 2)
if __name__ == "__main__":
    
    # Assuming API response has log probability fields named "logprobs" and "top_logprobs"
    model = VLMPreliminaryAnnotation(logprobs_field_name="logprobs", top_logprobs_field_name="top_logprobs", output_file="annotation_results.json")
    # Using relative paths
    test_images = [
        "./lambogini/Front.png",
        "./lambogini/Back.png",
        "./lambogini/Left.png",
        "./lambogini/Right.png",
        "./lambogini/Up.png",
        "./lambogini/Down.png"
    ]
    
    print("Starting image processing...")
    results = model(test_images)
    
    # Print result summary
    print("\nProcessing complete. Result summary:")
    for view, candidates in results.items():
        print(f"View {view}:")
        for i, (text, score) in enumerate(candidates[:2], 1):  # Only show first two candidate results
            print(f"  Candidate {i}: Confidence {score:.4f}")
            print(f"  Text: {text[:100]}..." if len(text) > 100 else f"  Text: {text}")
        if len(candidates) > 2:
            print(f"  ... {len(candidates)-2} more candidate results")
        print()
