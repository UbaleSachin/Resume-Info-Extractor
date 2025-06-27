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

from src.llm.resume_extractor import ResumeExtractor
from src.llm.candidate_fit import CandidateFitEvaluator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = FastAPI(title="Resume Extractor API", version="1.0.0")

extraction_jobs = {}
candidate_fit_jobs = {}
extraction_lock = threading.Lock()
fit_lock = threading.Lock()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the resume extractor
resume_extractor = ResumeExtractor()

class DownloadRequest(BaseModel):
    data: Dict[str, Any]
    format: str

class JobDescriptionRequest(BaseModel):
    job_description: str

class CandidateFitRequest(BaseModel):
    resume_data: List[Dict[str, Any]]
    job_description_data: str  # This should be string, not Dict

def process_extraction_job(job_id, files, resume_extractor):
    results = []
    batch_size = 15
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        for file_info in batch:
            file_path, filename = file_info
            try:
                result = resume_extractor.extract_from_file(file_path, filename)
                results.append(result)
            except Exception as e:
                results.append({"filename": filename, "error": str(e), "success": False})
    with extraction_lock:
        extraction_jobs[job_id]['status'] = 'completed'
        extraction_jobs[job_id]['results'] = results


@app.get("/")
async def root():
    return {"message": "Resume Extractor API is running"}

@app.post("/extract-resume")
async def extract_resume_data(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Extract data from uploaded resume files using Gemini API
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    job_id =str(uuid.uuid4())
    file_infos = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        file_infos.append((tmp_file_path, file.filename))
    with extraction_lock:
        extraction_jobs[job_id] = {'status': 'pending', 'results': None}
    background_tasks.add_task(process_extraction_job, job_id, file_infos, resume_extractor)
    return {'success': True, 'job_id': job_id, 'status': 'pending'}

@app.get("/extract-resume-/{job_id}")
async def get_extraction_status(job_id: str):
    with extraction_lock:
        job = extraction_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job['status'] == 'completed':
            return {'success': True, 'status': 'completed', 'extracted_data': job['results']}
        else:
            return {'success': True, 'status': job['status']}

@app.post("/extract-job-description")
async def extract_job_description(request: JobDescriptionRequest):
    """
    Process job description text using Gemini API
    """
    try:
        if not request.job_description or not request.job_description.strip():
            raise HTTPException(status_code=400, detail="Job description is required")
        
        # Here you would typically process the job description with your LLM
        # For now, we'll just return the processed text
        job_description_data = request.job_description.strip()
        
        return {
            "success": True,
            "job_description_data": job_description_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job description: {str(e)}")

def process_candidate_fit_job(job_id, resumes, job_description, evaluator):
    results = []
    batch_size = 15  # Gemini API limit
    for i in range(0, len(resumes), batch_size):
        batch = resumes[i:i+batch_size]
        for resume in batch:
            result = evaluator.evaluate_fit(resume, job_description)
            if result:
                result['candidate_name'] = resume.get('personal_info', {}).get('name', 'Unknown')
                results.append(result)
        if i + batch_size < len(resumes):
            time.sleep(60)
    with fit_lock:
        candidate_fit_jobs[job_id]['status'] = 'completed'
        candidate_fit_jobs[job_id]['results'] = results

@app.post("/candidate-fit")
async def candidate_fit(request: CandidateFitRequest, background_tasks: BackgroundTasks):
    """
    Compare multiple resumes and job description, return fit summary and percentage for each.
    """
    job_id = str(uuid.uuid4())
    with fit_lock:
        candidate_fit_jobs[job_id] = {'status': 'pending', 'results': None}
    background_tasks.add_task(process_candidate_fit_job, job_id, request.resume_data, request.job_description_data, CandidateFitEvaluator())
    return {"success": True, "job_id": job_id, "status": "pending"}

@app.get("/candidate-fit/{job_id}")
async def get_candidate_fit_job(job_id: str):
    with fit_lock:
        job = candidate_fit_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job['status'] == 'completed':
            return {"success": True, "status": "completed", "fit_results": job['results']}
        else:
            return {"success": True, "status": job['status']}
        
@app.post("/download-data")
async def download_data(request: DownloadRequest):
    """
    Convert extracted data to requested format and return as downloadable file
    """
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
                        #"Filename": resume.get("filename", ""),
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