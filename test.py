from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import csv
import pandas as pd
from io import BytesIO, StringIO
import tempfile
import os
import uuid
import threading
import time
import asyncio
from collections import deque, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from src.llm.resume_extractor import ResumeExtractor
from src.llm.candidate_fit import CandidateFitEvaluator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI(title="Resume Extractor API", version="2.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter for Gemini API (15 requests per minute)
class RateLimiter:
    def __init__(self, max_requests=15, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    async def acquire(self):
        """Wait until we can make a request within rate limits"""
        with self.lock:
            now = time.time()
            # Remove requests older than time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return
        
        # Need to wait
        sleep_time = self.time_window - (now - self.requests[0])
        await asyncio.sleep(sleep_time)
        await self.acquire()  # Try again

# In-memory job storage with TTL and cleanup
class InMemoryJobStorage:
    def __init__(self, ttl_seconds=3600):  # 1 hour TTL
        self.jobs = {}
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task to clean up expired jobs"""
        def cleanup():
            while True:
                self._cleanup_expired_jobs()
                time.sleep(300)  # Check every 5 minutes
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_jobs(self):
        """Remove expired jobs"""
        now = datetime.utcnow()
        with self.lock:
            expired_jobs = []
            for job_id, job_data in self.jobs.items():
                created_at = datetime.fromisoformat(job_data['created_at'])
                if (now - created_at).total_seconds() > self.ttl_seconds:
                    expired_jobs.append(job_id)
            
            for job_id in expired_jobs:
                del self.jobs[job_id]
    
    def create_job(self, job_id: str, job_type: str):
        job_data = {
            'status': 'pending',
            'results': None,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'type': job_type,
            'progress': 0
        }
        with self.lock:
            self.jobs[job_id] = job_data
    
    def update_job(self, job_id: str, status: str, results=None, progress=None):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                self.jobs[job_id]['updated_at'] = datetime.utcnow().isoformat()
                if results is not None:
                    self.jobs[job_id]['results'] = results
                if progress is not None:
                    self.jobs[job_id]['progress'] = progress
    
    def get_job(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id, None)
    
    def get_job_count(self):
        with self.lock:
            return len(self.jobs)

# Queue for managing concurrent jobs
class ProcessingQueue:
    def __init__(self, max_concurrent_jobs=5):
        self.queue = asyncio.Queue(maxsize=100)  # Limit queue size
        self.active_jobs = 0
        self.max_concurrent = max_concurrent_jobs
        self.lock = asyncio.Lock()
        self.worker_tasks = []
        self._start_workers()
    
    def _start_workers(self):
        """Start worker tasks to process the queue"""
        for _ in range(self.max_concurrent):
            task = asyncio.create_task(self._worker())
            self.worker_tasks.append(task)
    
    async def _worker(self):
        """Worker that processes jobs from the queue"""
        while True:
            try:
                job_data = await self.queue.get()
                if job_data is None:  # Shutdown signal
                    break
                
                async with self.lock:
                    self.active_jobs += 1
                
                # Determine job type and process accordingly
                job_type = job_data[0]
                if job_type == 'extraction':
                    await self._process_extraction_job(*job_data[1:])
                elif job_type == 'candidate_fit':
                    await self._process_candidate_fit_job(*job_data[1:])
                
                async with self.lock:
                    self.active_jobs -= 1
                
                self.queue.task_done()
            except Exception as e:
                print(f"Worker error: {e}")
                async with self.lock:
                    self.active_jobs -= 1
    
    async def add_job(self, job_type, *args):
        """Add a job to the processing queue"""
        try:
            await asyncio.wait_for(
                self.queue.put((job_type, *args)), 
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            return False  # Queue is full
    
    async def _process_extraction_job(self, job_id, files, resume_extractor):
        """Process a resume extraction job"""
        try:
            await process_extraction_job_async(job_id, files, resume_extractor)
        except Exception as e:
            job_storage.update_job(job_id, 'failed', {'error': str(e)})
    
    async def _process_candidate_fit_job(self, job_id, resumes, job_description, evaluator):
        """Process a candidate fit evaluation job"""
        try:
            await process_candidate_fit_job_async(job_id, resumes, job_description, evaluator)
        except Exception as e:
            job_storage.update_job(job_id, 'failed', {'error': str(e)})
    
    def get_stats(self):
        return {
            'active_jobs': self.active_jobs,
            'queue_size': self.queue.qsize(),
            'max_concurrent': self.max_concurrent
        }

# Global instances
rate_limiter = RateLimiter(max_requests=15, time_window=60)
job_storage = InMemoryJobStorage(ttl_seconds=3600)
executor = ThreadPoolExecutor(max_workers=10)
processing_queue = ProcessingQueue(max_concurrent_jobs=5)

# Initialize extractors
resume_extractor = ResumeExtractor()
candidate_fit_evaluator = CandidateFitEvaluator()

# Pydantic models
class DownloadRequest(BaseModel):
    data: Dict[str, Any]
    format: str

class JobDescriptionRequest(BaseModel):
    job_description: str

class CandidateFitRequest(BaseModel):
    resume_data: List[Dict[str, Any]]
    job_description_data: str

async def process_single_resume(resume_extractor, file_path, filename):
    """Process a single resume with rate limiting"""
    await rate_limiter.acquire()
    try:
        # Run the actual extraction in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            resume_extractor.extract_from_file, 
            file_path, 
            filename
        )
        return result
    except Exception as e:
        return {"filename": filename, "error": str(e), "success": False}

async def process_extraction_job_async(job_id, files, resume_extractor):
    """Process extraction job with progress tracking"""
    results = []
    total_files = len(files)
    
    # Update job status to processing
    job_storage.update_job(job_id, 'processing', progress=0)
    
    # Process files concurrently but respect rate limits
    semaphore = asyncio.Semaphore(3)  # Limit concurrent operations
    
    async def process_with_semaphore(file_info, index):
        async with semaphore:
            file_path, filename = file_info
            result = await process_single_resume(resume_extractor, file_path, filename)
            
            # Update progress
            progress = int((index + 1) * 100 / total_files)
            job_storage.update_job(job_id, 'processing', progress=progress)
            
            # Cleanup temp file
            try:
                os.unlink(file_path)
            except:
                pass
            
            return result
    
    # Create tasks for all files
    tasks = [
        process_with_semaphore(file_info, i) 
        for i, file_info in enumerate(files)
    ]
    
    # Process in batches to manage memory
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                results.append({"error": str(result), "success": False})
            else:
                results.append(result)
    
    # Mark job as completed
    job_storage.update_job(job_id, 'completed', results, progress=100)

async def process_single_candidate_fit(evaluator, resume, job_description):
    """Process a single candidate fit evaluation with rate limiting"""
    await rate_limiter.acquire()
    try:
        # Run the actual evaluation in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            evaluator.evaluate_fit, 
            resume, 
            job_description
        )
        if result:
            result['candidate_name'] = resume.get('personal_info', {}).get('name', 'Unknown')
        return result
    except Exception as e:
        candidate_name = resume.get('personal_info', {}).get('name', 'Unknown')
        return {"candidate_name": candidate_name, "error": str(e), "success": False}

async def process_candidate_fit_job_async(job_id, resumes, job_description, evaluator):
    """Process candidate fit job with progress tracking"""
    results = []
    total_resumes = len(resumes)
    
    # Update job status to processing
    job_storage.update_job(job_id, 'processing', progress=0)
    
    # Process resumes concurrently but respect rate limits
    semaphore = asyncio.Semaphore(3)  # Limit concurrent operations
    
    async def process_with_semaphore(resume, index):
        async with semaphore:
            result = await process_single_candidate_fit(evaluator, resume, job_description)
            
            # Update progress
            progress = int((index + 1) * 100 / total_resumes)
            job_storage.update_job(job_id, 'processing', progress=progress)
            
            return result
    
    # Create tasks for all resumes
    tasks = [
        process_with_semaphore(resume, i) 
        for i, resume in enumerate(resumes)
    ]
    
    # Process in batches to manage memory and rate limits
    batch_size = 15  # Gemini API batch size
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                results.append({"error": str(result), "success": False})
            elif result:  # Only add non-None results
                results.append(result)
    
    # Mark job as completed
    job_storage.update_job(job_id, 'completed', results, progress=100)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Resume Extractor API v2.0 is running with async processing"}

@app.post("/extract-resume")
async def extract_resume_data(files: List[UploadFile] = File(...)):
    """Extract data from uploaded resume files using Gemini API with async processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Limit number of files per request
    if len(files) > 500:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 500 files per request.")
    
    job_id = str(uuid.uuid4())
    file_infos = []
    
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            file_infos.append((tmp_file_path, file.filename))
        
        # Create job in storage
        job_storage.create_job(job_id, 'extraction')
        
        # Add to processing queue
        queued = await processing_queue.add_job('extraction', job_id, file_infos, resume_extractor)
        
        if not queued:
            # Clean up temp files if queue is full
            for file_path, _ in file_infos:
                try:
                    os.unlink(file_path)
                except:
                    pass
            raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")
        
        return {
            'success': True, 
            'job_id': job_id, 
            'status': 'queued',
            'estimated_time_minutes': len(files) // 15 + 1
        }
    
    except Exception as e:
        # Clean up temp files on error
        for file_path, _ in file_infos:
            try:
                os.unlink(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/extract-resume-/{job_id}")
async def get_extraction_status(job_id: str):
    """Get extraction job status with progress tracking"""
    job = job_storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        'success': True, 
        'status': job['status'],
        'progress': job.get('progress', 0),
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }
    
    if job['status'] == 'completed':
        response['extracted_data'] = job['results']
    elif job['status'] == 'failed':
        response['error'] = job.get('results', {}).get('error', 'Unknown error')
    
    return response

@app.post("/extract-job-description")
async def extract_job_description(request: JobDescriptionRequest):
    """Process job description text"""
    try:
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        job_description_data = request.job_description.strip()
        
        return {
            "success": True,
            "job_description_data": job_description_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job description: {str(e)}")

@app.post("/candidate-fit")
async def candidate_fit(request: CandidateFitRequest):
    """Compare multiple resumes and job description with async processing"""
    if not request.resume_data:
        raise HTTPException(status_code=400, detail="No resume data provided")
    
    if not request.job_description_data:
        raise HTTPException(status_code=400, detail="Job description is required")
    
    # Limit number of resumes per request
    if len(request.resume_data) > 100:
        raise HTTPException(status_code=400, detail="Too many resumes. Maximum 100 resumes per request.")
    
    job_id = str(uuid.uuid4())
    
    try:
        # Create job in storage
        job_storage.create_job(job_id, 'candidate_fit')
        
        # Add to processing queue
        queued = await processing_queue.add_job(
            'candidate_fit', 
            job_id, 
            request.resume_data, 
            request.job_description_data, 
            candidate_fit_evaluator
        )
        
        if not queued:
            raise HTTPException(status_code=503, detail="Server is busy. Please try again later.")
        
        return {
            "success": True, 
            "job_id": job_id, 
            "status": "queued",
            "estimated_time_minutes": len(request.resume_data) // 15 + 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidate-fit/{job_id}")
async def get_candidate_fit_job(job_id: str):
    """Get candidate fit job status with progress tracking"""
    job = job_storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        'success': True, 
        'status': job['status'],
        'progress': job.get('progress', 0),
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }
    
    if job['status'] == 'completed':
        response['fit_results'] = job['results']
    elif job['status'] == 'failed':
        response['error'] = job.get('results', {}).get('error', 'Unknown error')
    
    return response

@app.post("/download-data")
async def download_data(request: DownloadRequest):
    """Convert extracted data to requested format and return as downloadable file"""
    try:
        data = request.data
        format_type = request.format.lower()
        
        if format_type == "json":
            return create_json_response(data)
        elif format_type == "csv":
            return create_csv_response(data)
        elif format_type in ["excel", "xlsx"]:
            return create_excel_response(data)
        elif format_type == "pdf":
            return create_pdf_response(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

@app.get("/system/health")
async def health_check():
    """System health and statistics"""
    queue_stats = processing_queue.get_stats()
    return {
        "status": "healthy",
        "job_storage": {
            "total_jobs": job_storage.get_job_count(),
        },
        "processing_queue": queue_stats,
        "rate_limiter": {
            "requests_in_window": len(rate_limiter.requests)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/system/jobs")
async def list_jobs():
    """List all current jobs (for debugging)"""
    with job_storage.lock:
        jobs = {
            job_id: {
                'status': job_data['status'],
                'progress': job_data.get('progress', 0),
                'created_at': job_data['created_at'],
                'type': job_data['type']
            }
            for job_id, job_data in job_storage.jobs.items()
        }
    return {"jobs": jobs}

# File download helper functions (keeping original implementation)
def create_json_response(data):
    """Create JSON file response"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    return StreamingResponse(
        BytesIO(json_str.encode('utf-8')),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=resume_data.json"}
    )

def create_csv_response(data):
    """Create CSV file response"""
    output = StringIO()
    
    # Flatten the data for CSV format
    flattened_data = []
    
    if "extracted_data" in data:
        for resume in data["extracted_data"]:
            flat_resume = {}
            flat_resume["filename"] = resume.get("filename", "")
            
            # Extract basic info
            if "personal_info" in resume:
                for key, value in resume["personal_info"].items():
                    flat_resume[f"personal_{key}"] = value
            
            # Extract skills
            if "skills" in resume:
                if isinstance(resume["skills"], list):
                    flat_resume["skills"] = ", ".join(resume["skills"])
                else:
                    flat_resume["skills"] = str(resume["skills"])
            
            # Extract experience
            if "experience" in resume:
                for i, exp in enumerate(resume["experience"]):
                    if isinstance(exp, dict):
                        for key, value in exp.items():
                            flat_resume[f"experience_{i+1}_{key}"] = value
                    else:
                        flat_resume[f"experience_{i+1}"] = str(exp)
            
            # Extract education
            if "education" in resume:
                for i, edu in enumerate(resume["education"]):
                    if isinstance(edu, dict):
                        for key, value in edu.items():
                            flat_resume[f"education_{i+1}_{key}"] = value
                    else:
                        flat_resume[f"education_{i+1}"] = str(edu)
            
            flattened_data.append(flat_resume)
    
    if flattened_data:
        writer = csv.DictWriter(output, fieldnames=flattened_data[0].keys())
        writer.writeheader()
        writer.writerows(flattened_data)
    
    return StreamingResponse(
        BytesIO(output.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resume_data.csv"}
    )

def create_excel_response(data):
    """Create Excel file response in .xlsx format"""
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Create summary sheet
            summary_data = []
            
            if "extracted_data" in data:
                for resume in data["extracted_data"]:
                    summary_row = {
                        "Name": resume.get("personal_info", {}).get("name", ""),
                        "Email": resume.get("personal_info", {}).get("email", ""),
                        "Phone": resume.get("personal_info", {}).get("phone", ""),
                        "Skills": ", ".join(resume.get("skills", [])) if isinstance(resume.get("skills"), list) else str(resume.get("skills", "")),
                        "Experience": "; ".join(
                            f"{exp.get('title', '')} at {exp.get('company', '')}" 
                            for exp in resume.get("experience", []) if isinstance(exp, dict)
                        ) if isinstance(resume.get("experience", []), list) else "",
                        "Education": "; ".join(
                            f"{edu.get('degree', '')} from {edu.get('institution', '')}" 
                            for edu in resume.get("education", []) if isinstance(edu, dict)
                        ) if isinstance(resume.get("education", []), list) else "",
                        "Designation": resume.get("experience", [{}])[0].get("title", "") if isinstance(resume.get("experience"), list) and resume.get("experience") else "",
                        "Description": resume.get("experience", [{}])[0].get("description", "") if isinstance(resume.get("experience"), list) and resume.get("experience") else "",
                    }
                    summary_data.append(summary_row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create detailed sheet
            detailed_data = []
            if "extracted_data" in data:
                for resume in data["extracted_data"]:
                    detailed_row = {
                        "filename": resume.get("filename", ""),
                        "full_data": json.dumps(resume, indent=2, ensure_ascii=False)
                    }
                    detailed_data.append(detailed_row)
            
            if detailed_data:
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=resume_data.xlsx"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")

def create_pdf_response(data):
    """Create PDF report response"""
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Resume Extraction Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary
    if "extracted_data" in data:
        summary_text = f"Total Resumes Processed: {len(data['extracted_data'])}"
        summary = Paragraph(summary_text, styles['Heading2'])
        story.append(summary)
        story.append(Spacer(1, 12))
        
        # Individual resume details
        for i, resume in enumerate(data["extracted_data"], 1):
            # Resume header
            resume_title = Paragraph(f"Resume {i}: {resume.get('filename', 'Unknown')}", styles['Heading2'])
            story.append(resume_title)
            story.append(Spacer(1, 6))
            
            # Personal info
            if "personal_info" in resume:
                personal_info = resume["personal_info"]
                info_text = f"""
                Name: {personal_info.get('name', 'N/A')}<br/>
                Email: {personal_info.get('email', 'N/A')}<br/>
                Phone: {personal_info.get('phone', 'N/A')}<br/>
                Location: {personal_info.get('location', 'N/A')}
                """
                info_para = Paragraph(info_text, styles['Normal'])
                story.append(info_para)
                story.append(Spacer(1, 6))
            
            # Skills
            if "skills" in resume and resume["skills"]:
                skills_title = Paragraph("Skills:", styles['Heading3'])
                story.append(skills_title)
                if isinstance(resume["skills"], list):
                    skills_text = ", ".join(resume["skills"])
                else:
                    skills_text = str(resume["skills"])
                skills_para = Paragraph(skills_text, styles['Normal'])
                story.append(skills_para)
                story.append(Spacer(1, 6))
            
            # Experience
            if "experience" in resume and resume["experience"]:
                exp_title = Paragraph("Experience:", styles['Heading3'])
                story.append(exp_title)
                for exp in resume["experience"]:
                    if isinstance(exp, dict):
                        exp_text = f"{exp.get('title', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('duration', 'N/A')})"
                    else:
                        exp_text = str(exp)
                    exp_para = Paragraph(exp_text, styles['Normal'])
                    story.append(exp_para)
                story.append(Spacer(1, 6))
            
            # Education
            if "education" in resume and resume["education"]:
                edu_title = Paragraph("Education:", styles['Heading3'])
                story.append(edu_title)
                for edu in resume["education"]:
                    if isinstance(edu, dict):
                        edu_text = f"{edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})"
                    else:
                        edu_text = str(edu)
                    edu_para = Paragraph(edu_text, styles['Normal'])
                    story.append(edu_para)
                story.append(Spacer(1, 12))
    
    doc.build(story)
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume_report.pdf"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)