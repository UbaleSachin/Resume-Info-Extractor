from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, select
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import csv
import pandas as pd
import io
from io import BytesIO, StringIO
import tempfile
import os, asyncio
import uuid
import threading
import time
import asyncio
from collections import deque, defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import redis
from dotenv import load_dotenv

from src.llm.resume_extractor import ResumeExtractor
from src.llm.candidate_fit import CandidateFitEvaluator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

"""DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/yourdbname"  # Replace with your actual database URL
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")



class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True)
    hashed_password = Column(String(128))

async def create_db():
    async with engine.begin() as cnn:
        await cnn.run_sync(Base.metadata.create_all)"""

class UserIn(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

app = FastAPI(title="Resume Extractor API", version="2.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter for OpenAI API
class RateLimiter:
    def __init__(self, max_requests=450, time_window=60, max_tokens_per_minute=200000):  # Conservative limits
        self.max_requests = max_requests
        self.time_window = time_window
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests = deque()
        self.token_usage = deque()  # Track token usage with timestamps
        self.lock = asyncio.Lock()
        self.current_minute_tokens = 0
        self.last_minute_reset = time.time()
    
    async def acquire(self, estimated_tokens=1000):  # Default estimate for resume processing
        """Wait until we can make a request within both RPM and TPM limits"""
        async with self.lock:
            now = time.time()
            
            # Reset token counter every minute
            if now - self.last_minute_reset >= 60:
                self.current_minute_tokens = 0
                self.last_minute_reset = now
            
            # Remove old requests and token usage
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            while self.token_usage and self.token_usage[0][0] <= now - self.time_window:
                old_timestamp, old_tokens = self.token_usage.popleft()
                self.current_minute_tokens -= old_tokens
            
            # Check if we can make request within both limits
            if (len(self.requests) < self.max_requests and 
                self.current_minute_tokens + estimated_tokens <= self.max_tokens_per_minute):
                
                self.requests.append(now)
                self.token_usage.append((now, estimated_tokens))
                self.current_minute_tokens += estimated_tokens
                return
            
            # Calculate wait time based on the more restrictive limit
            rpm_wait = 0
            tpm_wait = 0
            
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                rpm_wait = self.time_window - (now - oldest_request) + 0.1
            
            if self.current_minute_tokens + estimated_tokens > self.max_tokens_per_minute:
                tpm_wait = 60 - (now - self.last_minute_reset) + 0.1
            
            wait_time = max(rpm_wait, tpm_wait)
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                await self.acquire(estimated_tokens)  # Retry after waiting

# Redis-based job storage
class JobStorage:
    def __init__(self):
        redis_pool = redis.ConnectionPool(
        host='localhost', 
        port=6379, 
        db=0, 
        max_connections=20,
        decode_responses=True
)
        self.redis_client = redis.Redis(connection_pool=redis_pool)
        self._update_cache = {}
        self._last_update = {}
    
    def create_job(self, job_id, job_type):
        job_data = {
            'status': 'pending',
            'results': None,
            'created_at': datetime.utcnow().isoformat(),
            'type': job_type,
            'progress': 0
        }
        self.redis_client.setex(f"job:{job_id}", 3600, json.dumps(job_data))
    
    def update_job(self, job_id, status, results=None, progress=None):
        # Batch progress updates to reduce Redis calls
        if status == 'processing' and progress is not None:
            now = time.time()
            if job_id in self._last_update and now - self._last_update[job_id] < 2:  # Minimum 2 seconds between updates
                return
            self._last_update[job_id] = now
        
        job_data = json.loads(self.redis_client.get(f"job:{job_id}") or '{}')
        job_data['status'] = status
        if results is not None:
            job_data['results'] = results
        if progress is not None:
            job_data['progress'] = progress
        job_data['updated_at'] = datetime.utcnow().isoformat()
        self.redis_client.setex(f"job:{job_id}", 3600, json.dumps(job_data))
    
    def get_job(self, job_id):
        job_data = self.redis_client.get(f"job:{job_id}")
        return json.loads(job_data) if job_data else None

# Updated ProcessingQueue with lower concurrency for GPT-4.1
class ProcessingQueue:
    def __init__(self, max_concurrent_jobs=15):  # Reduced from 15 to 8 for GPT-4.1
        self.queue = asyncio.Queue()
        self.active_jobs = 0
        self.max_concurrent = max_concurrent_jobs
        self.processing_task = None
        self.running = False
    
    async def add_job(self, job_type, job_id, *args):
        await self.queue.put((job_type, job_id, args))
        if not self.running:
            self.processing_task = asyncio.create_task(self._process_queue())
            self.running = True
        return True
    
    async def _process_queue(self):
        while True:
            try:
                # Check if we can process more jobs
                if self.active_jobs >= self.max_concurrent:
                    await asyncio.sleep(1.0)  # Increased sleep time for better stability
                    continue
                
                # Get next job with timeout
                job_type, job_id, args = await asyncio.wait_for(
                    self.queue.get(), timeout=2.0  # Increased timeout
                )
                
                self.active_jobs += 1
                
                # Process based on job type
                if job_type == 'extraction':
                    asyncio.create_task(self._process_extraction_job(job_id, *args))
                elif job_type == 'candidate_fit':
                    asyncio.create_task(self._process_candidate_fit_job(job_id, *args))
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in queue processing: {e}")
                continue
    
    async def _process_extraction_job(self, job_id, files, resume_extractor):
        try:
            await process_extraction_job_async(job_id, files, resume_extractor)
        except Exception as e:
            job_storage.update_job(job_id, 'failed', {'error': str(e)})
        finally:
            self.active_jobs -= 1

    async def _process_candidate_fit_job(self, job_id, resumes, job_description, evaluator, fit_options):
        try:
            await process_candidate_fit_job_async(job_id, resumes, job_description, evaluator, fit_options)
        except Exception as e:
            job_storage.update_job(job_id, 'failed', {'error': str(e)})
        finally:
            self.active_jobs -= 1

# Initialize extractors
resume_extractor = ResumeExtractor()
candidate_fit_evaluator = CandidateFitEvaluator()

# Updated global instances with new limits
rate_limiter = RateLimiter(max_requests=450, time_window=60, max_tokens_per_minute=200000)
job_storage = JobStorage()
executor = ThreadPoolExecutor(max_workers=30)  # Reduced from 30 to 20
processing_queue = ProcessingQueue(max_concurrent_jobs=15)  # Reduced from 15 to 8

# Pydantic models
class DownloadRequest(BaseModel):
    data: Dict[str, Any]
    format: str

class JobDescriptionRequest(BaseModel):
    job_description: str

class CandidateFitRequest(BaseModel):
    resume_data: List[Dict[str, Any]]
    job_description_data: str
    fit_options: Optional[Dict[str, Any]] = None

# Updated process_single_resume with token estimation
async def process_single_resume(resume_extractor, file_path, filename):
    """Process a single resume with rate limiting and token estimation"""
    # Estimate tokens based on file size (rough approximation)
    estimated_tokens = 1000  # Default estimate
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            # Rough estimation: 1 token per 4 characters, average resume ~4000 characters
            estimated_tokens = min(max(file_size // 4, 800), 2000)  # Between 800-2000 tokens
    except:
        estimated_tokens = 1000
    
    await rate_limiter.acquire(estimated_tokens)
    
    try:
        # Verify file exists before processing
        if not os.path.exists(file_path):
            return {"filename": filename, "error": "File not found", "success": False}
        
        # Run the actual extraction in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            resume_extractor.extract_from_file, 
            file_path, 
            filename
        )
        
        if result is None:
            print(f"Warning: No result for {filename}")
        
        return result
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {"filename": filename, "error": str(e), "success": False}
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up temp file {file_path}: {str(e)}")

# Updated process_extraction_job_async with better concurrency control
async def process_extraction_job_async(job_id, files, resume_extractor):
    """Process extraction job with progress tracking optimized for GPT-4.1"""
    try:
        results = []
        total_files = len(files)
        
        if total_files > 1:
            print(f"Starting extraction job {job_id} with {total_files} files")
        
        # Update job status to processing
        job_storage.update_job(job_id, 'processing', progress=0)
        
        # Reduced concurrency for GPT-4.1 TPM limits
        semaphore = asyncio.Semaphore(25)  # Reduced from 25 to 12
        
        async def process_with_semaphore(file_info, index):
            async with semaphore:
                file_path, filename = file_info
                result = await process_single_resume(resume_extractor, file_path, filename)
                
                # Update progress more frequently for better UX
                if total_files > 20:
                    # For large batches, update every 5% or at completion
                    if (index + 1) % max(1, total_files // 20) == 0 or index == total_files - 1:
                        progress = int((index + 1) * 100 / total_files)
                        job_storage.update_job(job_id, 'processing', progress=progress)
                else:
                    # For small batches, update every 2 files
                    if (index + 1) % 2 == 0 or index == total_files - 1:
                        progress = int((index + 1) * 100 / total_files)
                        job_storage.update_job(job_id, 'processing', progress=progress)
                
                return result
        
        # Create tasks for all files
        tasks = [
            process_with_semaphore(file_info, i) 
            for i, file_info in enumerate(files)
        ]
        
        # Process in smaller batches for better memory management and rate limiting
        batch_size = 50  # Reduced from 50 to 25
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"Task exception: {result}")
                    results.append({"error": str(result), "success": False})
                else:
                    results.append(result)
            
            # Add a small delay between batches to help with rate limiting
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.5)
        
        if total_files > 1 or any(r.get('error') for r in results):
            print(f"Extraction job {job_id} completed with {len(results)} results")
        
        # Mark job as completed
        job_storage.update_job(job_id, 'completed', results, progress=100)
        
    except Exception as e:
        print(f"Error in extraction job {job_id}: {str(e)}")
        job_storage.update_job(job_id, 'failed', {'error': str(e)})

# Updated process_single_candidate_fit with token estimation
async def process_single_candidate_fit(evaluator, resume, job_description, fit_options=None):
    """Process a single candidate fit evaluation with rate limiting and token estimation"""
    # Estimate tokens for candidate fit evaluation (usually higher than extraction)
    resume_text_length = len(str(resume))
    job_desc_length = len(job_description)
    estimated_tokens = min(max((resume_text_length + job_desc_length) // 3, 1200), 3000)
    
    await rate_limiter.acquire(estimated_tokens)
    
    try:
        # Run the actual evaluation in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            evaluator.evaluate_fit, 
            resume, 
            job_description,
            fit_options
        )
        if result:
            result['candidate_name'] = resume.get('personal_info', {}).get('name', 'Unknown')
        return result
    except Exception as e:
        candidate_name = resume.get('personal_info', {}).get('name', 'Unknown')
        return {"candidate_name": candidate_name, "error": str(e), "success": False}

# Updated process_candidate_fit_job_async with better concurrency control
async def process_candidate_fit_job_async(job_id, resumes, job_description, evaluator, fit_options=None):
    """Process candidate fit job with progress tracking optimized for GPT-4.1"""
    results = []
    total_resumes = len(resumes)
    
    # Update job status to processing
    job_storage.update_job(job_id, 'processing', progress=0)
    
    # Reduced concurrent operations for GPT-4.1 TPM limits
    semaphore = asyncio.Semaphore(20)  # Reduced from 20 to 10
    
    async def process_with_semaphore(resume, index):
        async with semaphore:
            result = await process_single_candidate_fit(evaluator, resume, job_description, fit_options)
            
            # Update progress more frequently
            if (index + 1) % 3 == 0 or index == total_resumes - 1:  # Every 3 resumes
                progress = int((index + 1) * 100 / total_resumes)
                job_storage.update_job(job_id, 'processing', progress=progress)
            
            return result
    
    # Create tasks for all resumes
    tasks = [
        process_with_semaphore(resume, i) 
        for i, resume in enumerate(resumes)
    ]
    
    # Process in smaller batches for candidate fit
    batch_size = 25  # Reduced from 25 to 15
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                results.append({"error": str(result), "success": False})
            elif result:  # Only add non-None results
                results.append(result)
        
        # Add a small delay between batches
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.3)
    
    # Mark job as completed
    job_storage.update_job(job_id, 'completed', results, progress=100)

"""@app.on_event("startup")
async def startup():
    await create_db()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(session, username: str):
    result = await session.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

async def get_user_by_id(session, user_id: int):
    #""Get user by ID - returns User object or None
    return await session.get(User, user_id)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Could not validate credentials", 
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    async with AsyncSessionLocal() as session:
        user = await get_user(session, username)
        if user is None:
            raise credentials_exception
        return user
    
@app.post("/register")
async def register(user: UserIn):
    async with AsyncSessionLocal() as session:
        db_user = await get_user(session, user.username)
        if db_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        hashed_password = get_password_hash(user.password)
        new_user = User(username=user.username, hashed_password=hashed_password)
        session.add(new_user)
        await session.commit()
        return {"message": "User registered successfully"}
    
@app.post("/login", response_model=Token)
async def login(form_data: UserIn):
    async with AsyncSessionLocal() as session:
        user = await get_user(session, form_data.username)
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token = create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}"""

@app.get("/")
async def root():
    return {"message": "Resume Extractor API v2.0 is running with async processing"}

# Updated extract_resume_data endpoint with better time estimation
@app.post("/extract-resume")
async def extract_resume_data(files: List[UploadFile] = File(...)):
    """
    Extract data from uploaded resume files using OpenAI API
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Limit number of files per request
        if len(files) > 1000:
            raise HTTPException(status_code=400, detail="Too many files. Maximum 1000 files per request.")
        
        job_id = str(uuid.uuid4())
        file_infos = []
        
        if len(files) > 1:
            print(f"Processing {len(files)} files for job {job_id}")
        
        for file in files:
            try:
                # Check file size and type
                if file.size > 10 * 1024 * 1024:  # 10MB limit
                    print(f"File {file.filename} too large: {file.size} bytes")
                    continue
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                file_infos.append((tmp_file_path, file.filename))

                if len(files) <= 5:
                    print(f"Saved temp file for {file.filename}: {tmp_file_path}")
                
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        if not file_infos:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        # Create job in storage
        job_storage.create_job(job_id, 'extraction')
        
        # Add to processing queue
        await processing_queue.add_job('extraction', job_id, file_infos, resume_extractor)

        # Updated time estimation for GPT-4.1 (slower due to TPM limits)
        estimated_time_minutes = max(2, len(file_infos) // 15)  # More conservative estimate
        
        return {
            'success': True, 
            'job_id': job_id, 
            'status': 'queued',
            'message': 'Job queued for processing',
            'estimated_time_minutes': estimated_time_minutes,
            'total_files': len(file_infos)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in extract_resume_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/extract-resume/{job_id}")
async def get_extraction_status(job_id: str):
    """Get job status with progress tracking"""
    try:
        job = job_storage.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Only log when job is completed or failed
        if job['status'] in ['completed', 'failed']:
            status_msg = f"Job {job_id} {job['status']}"
            if job['status'] == 'completed':
                result_count = len(job['results']) if job['results'] else 0
                status_msg += f" with {result_count} results"
            elif job['status'] == 'failed':
                status_msg += f": {job.get('results', {}).get('error', 'Unknown error')}"
            print(status_msg)
        
        response = {
            'success': True, 
            'status': job['status'],
            'progress': job.get('progress', 0),
            'created_at': job['created_at']
        }
        
        if 'updated_at' in job:
            response['updated_at'] = job['updated_at']
        
        if job['status'] == 'completed':
            response['extracted_data'] = job['results']
            print(f"Job {job_id} completed with {len(job['results']) if job['results'] else 0} results")
        elif job['status'] == 'failed':
            response['error'] = job.get('results', {}).get('error', 'Unknown error')
            print(f"Job {job_id} failed: {response['error']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting job status for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

# Updated candidate_fit endpoint with better limits and estimation
@app.post("/candidate-fit")
async def candidate_fit(request: CandidateFitRequest):
    """Compare multiple resumes and job description with async processing"""
    if not request.resume_data:
        raise HTTPException(status_code=400, detail="No resume data provided")
    
    if not request.job_description_data:
        raise HTTPException(status_code=400, detail="Job description is required")
    
    # Updated limit for GPT-4.1 (more conservative)
    if len(request.resume_data) > 500:  # Reduced from 500 to 300
        raise HTTPException(status_code=400, detail="Too many resumes. Maximum 300 resumes per request.")
    
    fit_options = request.fit_options or {}
    
    job_id = str(uuid.uuid4())
    
    try:
        # Create job in storage
        job_storage.create_job(job_id, 'candidate_fit')
        
        # Add to processing queue
        await processing_queue.add_job(
            'candidate_fit', 
            job_id, 
            request.resume_data, 
            request.job_description_data,
            candidate_fit_evaluator,
            fit_options
        )

        # Updated time estimation for GPT-4.1 (slower due to TPM limits)
        estimated_time_minutes = max(2, len(request.resume_data) // 10)  # More conservative
        
        return {
            "success": True, 
            "job_id": job_id, 
            "status": "queued",
            "message": "Job queued for processing",
            "estimated_time_minutes": estimated_time_minutes,
            "total_resumes": len(request.resume_data)
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
        'created_at': job['created_at']
    }
    
    if 'updated_at' in job:
        response['updated_at'] = job['updated_at']
    
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
    
@app.post("/download-fit-excel")
async def download_fit_excel(data: dict):
    fit_results = data.get("fit_results", [])
    if not fit_results:
        raise HTTPException(status_code=400, detail="No fit results provided.")

    # Flatten and prepare data for DataFrame
    rows = []
    for idx, fit in enumerate(fit_results, 1):
        rows.append({
            "Rank": idx,
            "Candidate Name": fit.get("candidate_name", f"Candidate {idx}"),
            "Fit Percentage": fit.get("fit_percentage", ""),
            "Summary": fit.get("summary", ""),
            "Key Matches": ", ".join(fit.get("key_matches", [])),
            "Key Gaps": ", ".join(fit.get("key_gaps", [])),
        })
    df = pd.DataFrame(rows)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=candidate_fit_results.xlsx"}
    )

"""@app.post("/save-extracted-data")
async def save_extracted_data(data: dict, user=Depends(get_current_user)):
    # Save data to a table with user_id reference (implement your own model)
    # Example: ResumeData(user_id=user.id, data=json.dumps(data))
    return {"msg": "Data saved for user", "user": user.username}

@app.get("/get-extracted-data")
async def get_extracted_data(user=Depends(get_current_user)):
    # Query your ResumeData table for user_id=user.id
    return {"msg": "Fetched data for user", "user": user.username}"""

# Updated health check to show TPM information
@app.get("/system/health")
async def health_check():
    """System health and statistics"""
    try:
        # Test resume extractor
        extractor_status = "unknown"
        try:
            extractor_status = "healthy" if resume_extractor else "not_initialized"
        except Exception as e:
            extractor_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "resume_extractor": extractor_status,
            "processing_queue": {
                "active_jobs": processing_queue.active_jobs,
                "max_concurrent": processing_queue.max_concurrent,
                "queue_size": processing_queue.queue.qsize()
            },
            "rate_limiter": {
                "requests_in_window": len(rate_limiter.requests),
                "max_requests": rate_limiter.max_requests,
                "time_window": rate_limiter.time_window,
                "current_rpm": len(rate_limiter.requests),
                "current_tpm": rate_limiter.current_minute_tokens,
                "max_tpm": rate_limiter.max_tokens_per_minute
            },
            "thread_pool": {
                "max_workers": executor._max_workers
            },
            "model_info": {
                "model": "gpt-4.1",
                "tpm_limit": 30000,
                "rpm_limit": 500,
                "tpd_limit": 900000
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# File download helper functions
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
                        "Experience": "; ".join([
                            f"{exp.get('title', '')} at {exp.get('company', '')} ({exp.get('location', '')}) - {exp.get('duration', '')}"
                            for exp in resume.get("experience", [])
                        ]),
                        "Education": "; ".join([
                            f"{edu.get('degree', '')} from {edu.get('institution', '')}"
                            for edu in resume.get("education", [])
                        ]),
                        "Designation": resume.get("experience", [{}])[0].get("title", "") if isinstance(resume.get("experience"), list) and resume.get("experience") else "",
                        "Summary": resume.get("summary", ""),
                        "Total Exeprience": resume.get("total_experience", ""),
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