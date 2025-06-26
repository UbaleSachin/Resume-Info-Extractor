import os
import json
import requests
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

class CandidateFitEvaluator:
    """
    Evaluates candidate fit for a job description using LLM.
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = 'https://generativelanguage.googleapis.com/v1beta'
        self.model = 'gemini-2.0-flash'
        self.prompt_template = """You are an expert HR recruiter and AI assistant. Given the following job description and candidate resume data, analyze and compare the candidate's qualifications, skills, and experience to the job requirements.

Return a JSON object with:
- "summary": A concise summary (2-4 sentences) explaining if the candidate is a good fit for the job, mentioning key matches and gaps.
- "fit_percentage": An approximate percentage (0-100) representing how well the candidate fits the job (e.g., 80 for strong fit, 40 for weak fit).
- "key_matches": List of main skills/requirements the candidate matches.
- "key_gaps": List of main skills/requirements the candidate lacks.

IMPORTANT: Only return the JSON object, no extra text.

Job Description:
{job_description}

Candidate Resume:
{resume}
"""
    
    def evaluate_fit(self, resume_data: Dict[str, Any], job_description_data: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Compares resume and job description using LLM and returns fit summary and percentage.
        
        Args:
            resume_data: Dictionary containing resume information
            job_description_data: Either a string or dictionary containing job description
        """
        # Handle both string and dict inputs for job description
        if isinstance(job_description_data, dict):
            job_desc_text = json.dumps(job_description_data, indent=2, ensure_ascii=False)
        else:
            job_desc_text = str(job_description_data)
        
        # Always convert resume to JSON string for consistency
        resume_text = json.dumps(resume_data, indent=2, ensure_ascii=False)
        
        prompt = self.prompt_template.format(
            job_description=job_desc_text,
            resume=resume_text
        )
        
        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {'key': self.gemini_api_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for more consistent results
                "maxOutputTokens": 1000
            }
        }
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        response_text = content['parts'][0]['text']
                        # Clean and parse JSON from response
                        json_str = self._extract_json(response_text)
                        try:
                            parsed_result = json.loads(json_str)
                            # Validate required fields
                            required_fields = ['summary', 'fit_percentage', 'key_matches', 'key_gaps']
                            if all(field in parsed_result for field in required_fields):
                                return parsed_result
                            else:
                                print(f"Missing required fields in response: {parsed_result}")
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON response: {json_str}, Error: {e}")
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("Request to Gemini API timed out")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
        except Exception as e:
            print(f"Error evaluating candidate fit: {str(e)}")
            
        return None
    
    def _extract_json(self, text: str) -> str:
        """
        Extracts JSON object from LLM response text.
        """
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        # If no JSON object found, return cleaned text
        return text.strip()