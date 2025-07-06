import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class CandidateFitEvaluator:
    """
    Evaluates candidate fit for a job description using LLM (Gemini or Groq).
    Rotates between models/providers based on API limits.
    """

    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

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
            },
            'openai': {
                'models': [('gpt-3.5-turbo', 10000),],
                'api_key': self.openai_api_key,
            }
        }
        self.usage_file = 'candidate_fit_api_usage.json'
        self.usage_data = self._load_usage_data()
        self.provider_order = ['openai']
        self.current_provider_idx = 0
        self.current_model_idx = 0

        self.prompt_template = """You are an expert HR recruiter and AI assistant. Given the following job description and candidate resume data, analyze and compare the candidate's qualifications, skills, and experience to the job requirements.

{custom_instructions}

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

    def _build_custom_instructions(self, fit_options: Optional[Dict[str, Any]]) -> str:
        """Build custom instructions based on fit_options"""
        if not fit_options:
            return ""
        
        instructions = []
        
        # Priority keywords
        if fit_options.get("priority_keywords"):
            keywords = fit_options["priority_keywords"]
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            instructions.append(f"Pay special attention to these priority keywords when evaluating fit: {keywords}")
            instructions.append(f"MANDATORY: Check each of these priority keywords [{keywords}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Required skills
        if fit_options.get("required_skills"):
            req_skills = fit_options["required_skills"]
            if isinstance(req_skills, list):
                req_skills = ", ".join(req_skills)
            instructions.append(f"Required skills: {req_skills}")
            instructions.append(f"MANDATORY: Check each required skill [{req_skills}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Minimum experience requirements
        if fit_options.get("min_experience"):
            min_exp = fit_options["min_experience"]
            instructions.append(f"The candidate must have at least {min_exp} years of relevant experience. Factor this heavily into your evaluation.")
            instructions.append(f"MANDATORY: If candidate has {min_exp}+ years experience, add 'Meets minimum experience requirement ({min_exp}+ years)' to 'key_matches'. If less, add 'Lacks minimum experience requirement ({min_exp} years)' to 'key_gaps'.")
        
        # Educational requirements
        if fit_options.get("edu_requirements"):
            edu_req = fit_options["edu_requirements"]
            instructions.append(f"Educational requirements: {edu_req}. Consider how well the candidate's education aligns with these requirements.")
            instructions.append(f"MANDATORY: Check if candidate meets education requirement '{edu_req}'. If yes, add to 'key_matches'. If no, add to 'key_gaps'.")
        
        # Weighting configuration
        weights = []
        if fit_options.get("weight_skills"):
            weights.append(f"Skills: {fit_options['weight_skills']}%")
        if fit_options.get("weight_experience"):
            weights.append(f"Experience: {fit_options['weight_experience']}%")
        if fit_options.get("weight_education"):
            weights.append(f"Education: {fit_options['weight_education']}%")
        
        if weights:
            instructions.append(f"Use the following weightage when calculating fit percentage - {', '.join(weights)}")
        
        # Deal breakers
        if fit_options.get("deal_breakers"):
            deal_breakers = fit_options["deal_breakers"]
            if isinstance(deal_breakers, list):
                deal_breakers = ", ".join(deal_breakers)
            instructions.append(f"These are deal breakers - if the candidate lacks any of these, significantly reduce the fit percentage: {deal_breakers}")
            instructions.append(f"MANDATORY: Check each deal breaker [{deal_breakers}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps' and reduce fit percentage significantly.")
        
        # Nice to have skills
        if fit_options.get("nice_to_have"):
            nice_to_have = fit_options["nice_to_have"]
            if isinstance(nice_to_have, list):
                nice_to_have = ", ".join(nice_to_have)
            instructions.append(f"These are nice-to-have skills that can boost the fit percentage: {nice_to_have}")
            instructions.append(f"MANDATORY: Check each nice-to-have skill [{nice_to_have}] against the candidate's resume. If found, add to 'key_matches' with '(nice-to-have)' notation.")
        
        # Specific experience areas
        if fit_options.get("experience_areas"):
            exp_areas = fit_options["experience_areas"]
            if isinstance(exp_areas, list):
                exp_areas = ", ".join(exp_areas)
            instructions.append(f"Required experience areas: {exp_areas}")
            instructions.append(f"MANDATORY: Check each experience area [{exp_areas}] against the candidate's background. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Technical skills
        if fit_options.get("technical_skills"):
            tech_skills = fit_options["technical_skills"]
            if isinstance(tech_skills, list):
                tech_skills = ", ".join(tech_skills)
            instructions.append(f"Required technical skills: {tech_skills}")
            instructions.append(f"MANDATORY: Check each technical skill [{tech_skills}] against the candidate's resume. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Certifications
        if fit_options.get("certifications"):
            certs = fit_options["certifications"]
            if isinstance(certs, list):
                certs = ", ".join(certs)
            instructions.append(f"Required/Preferred certifications: {certs}")
            instructions.append(f"MANDATORY: Check each certification [{certs}] against the candidate's credentials. If found, add to 'key_matches'. If missing, add to 'key_gaps'.")
        
        # Location preferences
        if fit_options.get("location_preference"):
            location = fit_options["location_preference"]
            instructions.append(f"Location preference: {location}. Consider geographical compatibility in your evaluation.")
        
        # Salary expectations
        if fit_options.get("salary_range"):
            salary = fit_options["salary_range"]
            instructions.append(f"Salary range: {salary}. Consider if the candidate's expectations align with this range.")
        
        if instructions:
            instructions.append(
                "IMPORTANT: For every priority keyword, required skill, experience, or education requirement, "
                "explicitly list each one in either 'key_matches' (if present) or 'key_gaps' (if missing). "
                "Do not skip any. Be exhaustive."
            )
            return "EVALUATION GUIDELINES:\n" + "\n".join(f"- {instruction}" for instruction in instructions) + "\n"
        
        return ""

    def evaluate_fit(self, resume_data: Dict[str, Any], job_description_data: Union[str, Dict[str, Any]], fit_options: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Evaluate candidate fit with optional customization parameters.
        
        Args:
            resume_data: Dictionary containing candidate resume information
            job_description_data: Job description as string or dictionary
            fit_options: Optional dictionary with evaluation parameters:
                - priority_keywords: List of important keywords to prioritize
                - required_skills: List of mandatory skills
                - min_experience: Minimum years of experience required
                - edu_requirements: Educational requirements description
                - weight_skills: Percentage weight for skills (0-100)
                - weight_experience: Percentage weight for experience (0-100)  
                - weight_education: Percentage weight for education (0-100)
                - deal_breakers: List of must-have requirements
                - nice_to_have: List of preferred but not required skills
                - experience_areas: List of required experience areas
                - technical_skills: List of required technical skills
                - certifications: List of required/preferred certifications
                - location_preference: Location requirements
                - salary_range: Expected salary range
            max_retries: Maximum number of API call retries
            
        Returns:
            Dictionary with evaluation results or None if failed
        """
        # Prepare job description text
        if isinstance(job_description_data, dict):
            job_desc_text = json.dumps(job_description_data, indent=2, ensure_ascii=False)
        else:
            job_desc_text = str(job_description_data)
        
        # Prepare resume text
        resume_text = json.dumps(resume_data, indent=2, ensure_ascii=False)
        
        # Build custom instructions based on fit_options
        custom_instructions = self._build_custom_instructions(fit_options)
        
        # Create the prompt
        prompt = self.prompt_template.format(
            custom_instructions=custom_instructions,
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

    def _make_openai_api_call(self, text_content: str, model: str) -> Optional[str]:
        """Make API call to openAI for resume extraction."""
        try:
            full_prompt = self.extraction_prompt + text_content
            client = OpenAI(api_key = self.openai_api_key)
            completion = client.chat.completions.create(
                model = model,
                messages = [
                    {
                        "role": "user", "content": full_prompt,
                    }
                ],
                max_tokens = 2000,
                temperature = 0  # Lower temperature for more consistent JSON output
            )

            # upadte usage tracking
            self._update_api_usage('openai', model, 1)
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error making OpenAI API call: {str(e)}")
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