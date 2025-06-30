import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

class CandidateFitEvaluator:
    """
    Evaluates candidate fit for a job description using LLM (Gemini or Groq).
    Rotates between models/providers based on API limits.
    """

    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')

        self.providers = {
            'gemini': {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'models': [
                    ('gemma-3-27b-it', 14400),
                    ('gemma-3-12b-it', 14400),
                    ('gemma-3n-e4b-it', 14400),
                    ('gemma-3n-e2b-it', 14400),
                    ('gemini-2.0-flash-lite', 1500),
                    ('gemini-2.0-flash', 1500),
                    ('gemini-2.5-flash-preview-04-17', 500),
                ],
                'api_key': self.gemini_api_key
            },
            'groq': {
                'base_url': 'https://api.groq.com/openai/v1',
                'models': [
                    ('llama3-70b-8192', 14400),
                    ('llama3-8b-8192', 14400),
                    ('llama-3.3-70b-versatile', 1000),
                    ('mixtral-saba-24b', 1000),
                ],
                'api_key': self.groq_api_key
            }
        }
        self.usage_file = 'candidate_fit_api_usage.json'
        self.usage_data = self._load_usage_data()
        self.provider_order = ['gemini', 'groq']
        self.current_provider_idx = 0
        self.current_model_idx = 0

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

    def _load_usage_data(self):
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception:
            return {}

    def _save_usage_data(self):
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception:
            pass

    def _get_today_key(self):
        return datetime.now().strftime('%Y-%m-%d')

    def _update_api_usage(self, provider, model, count=1):
        today = self._get_today_key()
        if provider not in self.usage_data:
            self.usage_data[provider] = {}
        if model not in self.usage_data[provider]:
            self.usage_data[provider][model] = {}
        if today not in self.usage_data[provider][model]:
            self.usage_data[provider][model][today] = 0
        self.usage_data[provider][model][today] += count
        self._save_usage_data()

    def _get_today_usage(self, provider, model):
        today = self._get_today_key()
        return self.usage_data.get(provider, {}).get(model, {}).get(today, 0)

    def _can_use_model(self, provider, model, daily_limit):
        usage = self._get_today_usage(provider, model)
        return usage < daily_limit

    def _switch_to_next_model(self):
        # Try next model in current provider
        provider = self.provider_order[self.current_provider_idx]
        models = self.providers[provider]['models']
        if self.current_model_idx < len(models) - 1:
            self.current_model_idx += 1
            return True
        # Try next provider
        for idx in range(len(self.provider_order)):
            next_provider_idx = (self.current_provider_idx + 1 + idx) % len(self.provider_order)
            next_provider = self.provider_order[next_provider_idx]
            if self.providers[next_provider]['api_key']:
                self.current_provider_idx = next_provider_idx
                self.current_model_idx = 0
                return True
        return False

    def evaluate_fit(self, resume_data: Dict[str, Any], job_description_data: Union[str, Dict[str, Any]], max_retries=3) -> Optional[Dict[str, Any]]:
        # Prepare prompt
        if isinstance(job_description_data, dict):
            job_desc_text = json.dumps(job_description_data, indent=2, ensure_ascii=False)
        else:
            job_desc_text = str(job_description_data)
        resume_text = json.dumps(resume_data, indent=2, ensure_ascii=False)
        prompt = self.prompt_template.format(
            job_description=job_desc_text,
            resume=resume_text
        )

        for attempt in range(max_retries):
            provider = self.provider_order[self.current_provider_idx]
            provider_conf = self.providers[provider]
            model, daily_limit = provider_conf['models'][self.current_model_idx]
            api_key = provider_conf['api_key']

            if not api_key or not self._can_use_model(provider, model, daily_limit):
                if not self._switch_to_next_model():
                    break
                continue

            if provider == 'gemini':
                result = self._call_gemini(prompt, model, api_key)
            elif provider == 'groq':
                result = self._call_groq(prompt, model, api_key)
            else:
                result = None

            if result:
                self._update_api_usage(provider, model, 1)
                return result
            else:
                if not self._switch_to_next_model():
                    break
        return None

    def _call_gemini(self, prompt, model, api_key):
        url = f"{self.providers['gemini']['base_url']}/models/{model}:generateContent"
        params = {'key': api_key}
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
                "temperature": 0.1,
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
                        json_str = self._extract_json(response_text)
                        parsed_result = json.loads(json_str)
                        required_fields = ['summary', 'fit_percentage', 'key_matches', 'key_gaps']
                        if all(field in parsed_result for field in required_fields):
                            return parsed_result
            return None
        except Exception:
            return None

    def _call_groq(self, prompt, model, api_key):
        url = f"{self.providers['groq']['base_url']}/chat/completions"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 1000,
            'temperature': 0.1
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    response_text = result['choices'][0]['message']['content']
                    json_str = self._extract_json(response_text)
                    parsed_result = json.loads(json_str)
                    required_fields = ['summary', 'fit_percentage', 'key_matches', 'key_gaps']
                    if all(field in parsed_result for field in required_fields):
                        return parsed_result
            return None
        except Exception:
            return None

    def _extract_json(self, text: str) -> str:
        import re
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        return text.strip()
