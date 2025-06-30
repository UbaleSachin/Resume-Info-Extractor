class ResumeExtractor {
    constructor() {
        this.selectedFiles = [];
        this.extractedData = null;
        this.jobDescriptionData = null;
        this.initializeEventListeners();
        this.initializeJobDescriptionEvents();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const extractBtn = document.getElementById('extractBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const removeAllBtn = document.getElementById('removeAllBtn');
        const generateBtn = document.getElementById('generateBtn');

        if (generateBtn) {
            generateBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                // Call both extractData and extractJobDescription in sequence
                await this.extractData();
                await this.extractJobDescription();

                if (this.extractedData && this.jobDescriptionData) {
                    this.fetchCandidateFit();
                }
            });
        }

        // Prevent file dialog from reopening after cancel
        if (fileInput) {
            fileInput.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }

        // Only open file dialog on direct click, not on drop
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', (e) => {
                if (e.target === fileInput) return; // Already handled
                fileInput.value = ''; // Reset so selecting the same file twice works
                fileInput.click();
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // Drag and drop events
        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        }

        // Button events
        if (extractBtn) {
            extractBtn.addEventListener('click', () => this.extractData());
        }
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadData());
        }
        if (removeAllBtn) {
            removeAllBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // Prevent click from bubbling to uploadArea
                this.removeAllFiles();
            });
        }
    }

    initializeJobDescriptionEvents() {
        const jdUploadArea = document.getElementById('jdUploadArea');
        const jdInput = document.getElementById('jdInput');

        // Only initialize if elements exist
        if (!jdUploadArea || !jdInput) {
            console.warn('Job description elements not found. Make sure you have elements with IDs "jdUploadArea" and "jdInput" in your HTML.');
            return;
        }

        // Create file input for job description
        this.createJobDescriptionFileInput();

        // Create preview element if it doesn't exist
        this.createJobDescriptionPreview();

        const jdFileInput = document.getElementById('jdFileInput');

        // Drag and drop for job description
        jdUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            jdUploadArea.classList.add('dragover');
        });

        jdUploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Only remove dragover if we're actually leaving the drop area
            if (!jdUploadArea.contains(e.relatedTarget)) {
                jdUploadArea.classList.remove('dragover');
            }
        });

        jdUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            jdUploadArea.classList.remove('dragover');
            
            // Try to get text from dropped files or plain text
            if (e.dataTransfer.items) {
                for (let i = 0; i < e.dataTransfer.items.length; i++) {
                    const item = e.dataTransfer.items[i];
                    if (item.kind === 'string' && item.type === 'text/plain') {
                        item.getAsString((str) => {
                            jdInput.value = str;
                            this.updateJobDescriptionPreview();
                        });
                        return;
                    } else if (item.kind === 'file') {
                        const file = item.getAsFile();
                        if (file && (file.type.startsWith('text/') || file.name.toLowerCase().endsWith('.txt') || file.name.toLowerCase().endsWith('.docx') || file.name.toLowerCase().endsWith('.doc'))) {
                            this.readJobDescriptionFile(file);
                            return;
                        }
                    }
                }
            }
            
            // Fallback: try to get plain text
            const text = e.dataTransfer.getData('text/plain');
            if (text) {
                jdInput.value = text;
                this.updateJobDescriptionPreview();
            }
        });

        // Click to open file dialog or focus textarea
        jdUploadArea.addEventListener('click', (e) => {
            // If clicking on textarea, let it focus normally
            if (e.target === jdInput) {
                return;
            }
            
            // If clicking on the upload area but not on textarea, open file dialog
            if (jdFileInput && !jdInput.contains(e.target)) {
                jdFileInput.value = ''; // Reset so selecting the same file twice works
                jdFileInput.click();
            }
        });

        // Handle file selection from file input
        if (jdFileInput) {
            jdFileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.readJobDescriptionFile(file);
                }
            });
        }

        // Listen for input changes
        jdInput.addEventListener('input', () => {
            this.updateJobDescriptionPreview();
        });

        // Listen for paste events
        jdInput.addEventListener('paste', () => {
            setTimeout(() => this.updateJobDescriptionPreview(), 10);
        });

        // Initialize preview
        this.updateJobDescriptionPreview();
    }

    createJobDescriptionFileInput() {
        // Check if file input already exists
        let jdFileInput = document.getElementById('jdFileInput');
        if (!jdFileInput) {
            jdFileInput = document.createElement('input');
            jdFileInput.type = 'file';
            jdFileInput.id = 'jdFileInput';
            jdFileInput.accept = '.txt,.doc,.docx,.pdf';
            jdFileInput.style.display = 'none';
            document.body.appendChild(jdFileInput);
        }
    }

    readJobDescriptionFile(file) {
        const jdInput = document.getElementById('jdInput');
        if (!jdInput) return;

        // For .txt files, read directly
        if (file.type === 'text/plain' || file.name.toLowerCase().endsWith('.txt')) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                jdInput.value = evt.target.result;
                this.updateJobDescriptionPreview();
            };
            reader.readAsText(file);
        } 
        // For other file types, show message
        else {
            this.showError(`File "${file.name}" format not fully supported for job descriptions. Please use .txt files or copy/paste the content directly.`);
        }
    }

    createJobDescriptionPreview() {
        const jdUploadArea = document.getElementById('jdUploadArea');
        if (!jdUploadArea) return;

        // Check if preview already exists
        let jdPreview = document.getElementById('jdPreview');
        if (!jdPreview) {
            jdPreview = document.createElement('div');
            jdPreview.id = 'jdPreview';
            jdPreview.style.cssText = `
                display: none;
                margin-top: 15px;
                padding: 12px;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                font-size: 0.9rem;
            `;
            jdUploadArea.appendChild(jdPreview);
        }
    }


    removeAllFiles() {
        this.selectedFiles = [];
        this.displaySelectedFiles();
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.add('dragover');
        }
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.remove('dragover');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea) {
            uploadArea.classList.remove('dragover');
        }
        
        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }

    processFiles(files) {
        // Filter supported file types
        const supportedTypes = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain'
        ];

        const validFiles = files.filter(file => {
            const isValid = supportedTypes.includes(file.type) || 
                           file.name.toLowerCase().endsWith('.pdf') ||
                           file.name.toLowerCase().endsWith('.doc') ||
                           file.name.toLowerCase().endsWith('.docx') ||
                           file.name.toLowerCase().endsWith('.xls') ||
                           file.name.toLowerCase().endsWith('.xlsx') ||
                           file.name.toLowerCase().endsWith('.txt');
            
            if (!isValid) {
                this.showError(`File "${file.name}" is not supported. Please upload PDF, DOC, DOCX, XLS, XLSX, or TXT files.`);
            }
            return isValid;
        });

        if (validFiles.length > 0) {
            this.selectedFiles = [...this.selectedFiles, ...validFiles];
            this.displaySelectedFiles();
            this.hideError();
        }
    }

    displaySelectedFiles() {
        const filesPreview = document.getElementById('filesPreview');
        const filesList = document.getElementById('filesList');
        
        if (!filesPreview || !filesList) return;
        
        if (this.selectedFiles.length === 0) {
            filesPreview.style.display = 'none';
            return;
        }

        filesPreview.style.display = 'block';
        filesPreview.classList.add('fade-in');
        
        filesList.innerHTML = '';
        
        this.selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item slide-in';
            fileItem.style.animationDelay = `${index * 0.1}s`;
            
            fileItem.innerHTML = `
                <div class="file-info">
                    <div class="file-icon">${this.getFileIcon(file.name)}</div>
                    <div class="file-details">
                        <h4>${file.name}</h4>
                        <p>${this.formatFileSize(file.size)} • ${file.type || 'Unknown type'}</p>
                    </div>
                </div>
                <button class="remove-file" type="button">×</button>
            `;

            // Attach event listener to the remove button
            const removeBtn = fileItem.querySelector('.remove-file');
            if (removeBtn) {
                removeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.removeFile(e, index);
                });
            }
            
            filesList.appendChild(fileItem);
        });
    }

    removeFile(event, index) {
        if (event) event.stopPropagation();
        this.selectedFiles.splice(index, 1);
        this.displaySelectedFiles();
    }

    getFileIcon(filename) {
        const extension = filename.toLowerCase().split('.').pop();
        const icons = {
            'pdf': '📄',
            'doc': '📝',
            'docx': '📝',
            'xls': '📊',
            'xlsx': '📊',
            'txt': '📄'
        };
        return icons[extension] || '📄';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async extractData() {
        if (this.selectedFiles.length === 0) {
            this.showError('Please select at least one file to extract data.');
            return;
        }

        this.showLoading();
        this.hideError();

        try {
            const formData = new FormData();
            this.selectedFiles.forEach((file, index) => {
                formData.append('files', file);
            });
            
            const response = await fetch('http://localhost:8000/extract-resume', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!data.success || !data.job_id) {
                throw new Error('Failed to start extraction job.');
            }
            const jobId = data.job_id;

            // Poll for extraction status
            let extractedData = null;
            while (true) {
                await new Promise(res => setTimeout(res, 3000)); // wait 3 seconds
                const pollResp = await fetch(`http://localhost:8000/extract-resume-/${jobId}`);
                const pollData = await pollResp.json();
                if (pollData.status === 'completed') {
                    extractedData = pollData;
                    break;
                }
                // Optionally, show progress to user here
            }
            this.extractedData = extractedData;
            this.displayResults(extractedData);
            
        } catch (error) {
           console.error('Error extracting data:', error);
           this.showError('Failed to extract data. Please make sure the server is running and try again.');
        } finally {
            this.hideLoading();
        }
    }

    async extractJobDescription() {
        const jdInput = document.getElementById('jdInput');
        if (!jdInput || !jdInput.value.trim()) {
            this.showError('Please enter or drop a job description.');
            return;
        }

        this.showLoading();
        this.hideError();

        try {
            const jobDescriptionText = jdInput.value.trim();
            // Optionally, you can still call your backend for JD extraction if needed
            // But for candidate fit, store the string
            this.jobDescriptionData = jobDescriptionText;

            // If you want to call /extract-job-description, do it here, but don't overwrite jobDescriptionData with the response object

            // this.showSuccess('Job description processed successfully!');
        } catch (error) {
            console.error('Error extracting job description:', error);
            this.showError('Failed to extract job description. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async fetchCandidateFit() {
        if (!this.extractedData || !this.jobDescriptionData) {
            this.showError('Please extract both resume data and job description first.');
            return;
        }
        this.hideError();

        try {
            // Send all resumes for evaluation
            const resumes = this.extractedData.extracted_data || [];
            const response = await fetch('http://localhost:8000/candidate-fit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    resume_data: resumes,
                    job_description_data: this.jobDescriptionData
                })
            });
            const data = await response.json();
            if (!data.success || !data.job_id) {
                throw new Error('Failed to start candidate fit job.');
            }
            const jobId = data.job_id;

            // Poll for candidate fit status
            let fitData = null;
            while (true) {
                await new Promise(res => setTimeout(res, 3000)); // wait 3 seconds
                const pollResp = await fetch(`http://localhost:8000/candidate-fit/${jobId}`);
                const pollData = await pollResp.json();
                if (pollData.status === 'completed') {
                    fitData = pollData;
                    break;
                }
                // Optionally, show progress to user here
            }
            this.displayCandidateFit(fitData);
        } catch (error) {
            this.showError(`Failed to fetch candidate fit summary: ${error.message}`);
        }
    }

    displayCandidateFit(data) {
        let candidateFitSection = document.getElementById('candidateFitSection');
        if (!candidateFitSection) {
            const resultsSection = document.getElementById('resultsSection');
            candidateFitSection = document.createElement('div');
            candidateFitSection.id = 'candidateFitSection';
            candidateFitSection.className = 'candidate-fit-section';
            candidateFitSection.style.marginBottom = '30px';
            candidateFitSection.innerHTML = `
                <h3 style="margin-bottom: 15px;">Candidate Fit Analysis</h3>
                <div id="candidateFitContent"></div>
            `;
            const downloadOptions = document.querySelector('.download-options');
            if (resultsSection && downloadOptions) {
                resultsSection.insertBefore(candidateFitSection, downloadOptions);
            } else if (resultsSection) {
                resultsSection.prepend(candidateFitSection);
            }
        }

        const fitContent = document.getElementById('candidateFitContent');
        if (!fitContent) return;

        // Support multiple results
        const fitResults = data && (data.fit_results || (data.fit_result && [data.fit_result]));
        if (fitResults && fitResults.length > 0) {
            let html = `<div class="fit-list${fitResults.length > 10 ? ' fit-list-scroll' : ''}">`;
            fitResults.forEach((fitResult, idx) => {
                const candidateName = fitResult.candidate_name || `Candidate ${idx + 1}`;
                html += `
                    <div class="fit-item" style="margin-bottom:18px;">
                        <div class="fit-header" style="display:flex;align-items:center;cursor:pointer;gap:18px;padding:12px 0;" data-fit-index="${idx}">
                            <span style="font-weight:700;font-size:1.1rem;">${candidateName}</span>
                            <span class="fit-score-badge" style="margin-left:auto;background:#e9ecef;padding:6px 16px;border-radius:16px;font-weight:600;color:#333;">
                                ${fitResult.fit_percentage}%
                            </span>
                            <span class="expand-arrow" style="margin-left:10px;transition:transform 0.2s;">&#9654;</span>
                        </div>
                        <div class="fit-details" style="display:none;padding:12px 0 0 0;">
                            <div class="fit-summary" style="margin-bottom:12px;">${fitResult.summary || ''}</div>
                            ${fitResult.key_matches && fitResult.key_matches.length > 0 ? `
                                <div class="fit-matches" style="margin-bottom:10px;">
                                    <strong>✓ Key Matches</strong>
                                    <ul>
                                        ${fitResult.key_matches.map(match => `<li>${match}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            ${fitResult.key_gaps && fitResult.key_gaps.length > 0 ? `
                                <div class="fit-gaps">
                                    <strong>⚠ Key Gaps</strong>
                                    <ul>
                                        ${fitResult.key_gaps.map(gap => `<li>${gap}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            });
            html += `</div>`;
            fitContent.innerHTML = html;

            // Add expand/collapse logic
            fitContent.querySelectorAll('.fit-header').forEach(header => {
                header.addEventListener('click', function () {
                    const details = this.parentElement.querySelector('.fit-details');
                    const arrow = this.querySelector('.expand-arrow');
                    if (details.style.display === 'none' || !details.style.display) {
                        details.style.display = 'block';
                        arrow.style.transform = 'rotate(90deg)';
                    } else {
                        details.style.display = 'none';
                        arrow.style.transform = '';
                    }
                });
            });
        } else {
            fitContent.innerHTML = '<p style="color: #dc3545;">No candidate fit analysis available.</p>';
        }
    }

    // Helper function to determine the background position for the gradient bar
    getFitScoreBarPosition(percentage) {
        if (percentage <= 25) return 0;
        if (percentage <= 50) return 25;
        if (percentage <= 75) return 50;
        return 75;
    }

    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const dataPreview = document.getElementById('dataPreview');
        
        if (!resultsSection || !dataPreview) return;
        
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Show a readable preview of each resume
        if (data && data.extracted_data && data.extracted_data.length > 0) {
            // Build a table preview with correct column order
            let table = `<div style="overflow-x: auto;">
                <table style="width:100%;border-collapse:collapse;margin-top:10px;">
                    <thead>
                        <tr style="background-color:#f5f5f5;">
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">#</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Name</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Email</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Phone</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Location</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Designation</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Skills</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Education</th>
                            <th style="border:1px solid #ddd;padding:8px;text-align:left;">Experience</th>
                        </tr>
                    </thead>
                    <tbody>`;
            
            data.extracted_data.forEach((resume, idx) => {
                const name = resume.personal_info?.name || '';
                const email = resume.personal_info?.email || '';
                const phone = resume.personal_info?.phone || '';
                const location = resume.personal_info?.location || '';
                const designation = resume.personal_info?.designation || '';
                
                const skills = Array.isArray(resume.skills) 
                    ? resume.skills.join(', ') 
                    : (resume.skills || '');
                
                const education = Array.isArray(resume.education)
                    ? resume.education.map(e => {
                        if (typeof e === 'object') {
                            return Object.values(e).filter(v => v).join(', ');
                        }
                        return e;
                    }).join(' | ')
                    : (resume.education || '');
                
                const experience = Array.isArray(resume.experience)
                    ? resume.experience.map(e => {
                        if (typeof e === 'object') {
                            return Object.values(e).filter(v => v).join(', ');
                        }
                        return e;
                    }).join(' | ')
                    : (resume.experience || '');

                const description = resume.experience?.description || '';
                
                table += `<tr>
                    <td style="border:1px solid #ddd;padding:8px;">${idx + 1}</td>
                    <td style="border:1px solid #ddd;padding:8px;">${name}</td>
                    <td style="border:1px solid #ddd;padding:8px;">${email}</td>
                    <td style="border:1px solid #ddd;padding:8px;">${phone}</td>
                    <td style="border:1px solid #ddd;padding:8px;">${location}</td>
                    <td style="border:1px solid #ddd;padding:8px;">${designation}</td>
                    <td style="border:1px solid #ddd;padding:8px;max-width:200px;word-wrap:break-word;">${skills}</td>
                    <td style="border:1px solid #ddd;padding:8px;max-width:200px;word-wrap:break-word;">${education}</td>
                    <td style="border:1px solid #ddd;padding:8px;max-width:200px;word-wrap:break-word;">${experience}</td>
                    <td style="border:1px solid #ddd;padding:8px;max-width:200px;word-wrap:break-word;">${designation}</td>
                </tr>`;
            });
            table += '</tbody></table></div>';
            dataPreview.innerHTML = table;
        } else {
            dataPreview.innerHTML = '<em>No data extracted.</em>';
        }

        // Only fetch candidate fit if we have job description data
        if (this.jobDescriptionData) {
            this.fetchCandidateFit();
        }
    }

    async downloadData() {
        if (!this.extractedData) {
            this.showError('No data to download. Please extract data first.');
            return;
        }

        const format = 'xlsx';
        
        try {
            const response = await fetch('http://localhost:8000/download-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: this.extractedData,
                    format: format
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Get the filename from the response headers
            const contentDisposition = response.headers.get('content-disposition');
            let filename = `resume_data.${format}`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }

            // Create blob and download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

        } catch (error) {
            console.error('Error downloading data:', error);
            this.showError('Failed to download data. Please try again.');
        }
    }

    // Add success message method
    showSuccess(message) {
        // Create success element if it doesn't exist
        let successMessage = document.getElementById('successMessage');
        if (!successMessage) {
            successMessage = document.createElement('div');
            successMessage.id = 'successMessage';
            successMessage.style.cssText = `
                display: none;
                margin: 10px 0;
                padding: 12px;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 4px;
                color: #155724;
            `;
            // Insert after error message if it exists
            const errorMessage = document.getElementById('errorMessage');
            if (errorMessage) {
                errorMessage.parentNode.insertBefore(successMessage, errorMessage.nextSibling);
            } else {
                document.body.appendChild(successMessage);
            }
        }
        
        successMessage.textContent = message;
        successMessage.style.display = 'block';
        
        // Auto-hide success after 3 seconds
        setTimeout(() => {
            successMessage.style.display = 'none';
        }, 3000);
    }

    showLoading() {
        const loading = document.getElementById('loading');
        const extractBtn = document.getElementById('extractBtn');
        const removeAllBtn = document.getElementById('removeAllBtn');
        
        if (loading) loading.style.display = 'block';
        if (extractBtn) extractBtn.style.display = 'none';
        if (removeAllBtn) removeAllBtn.disabled = true;
    }

    hideLoading() {
        const loading = document.getElementById('loading');
        const extractBtn = document.getElementById('extractBtn');
        const removeAllBtn = document.getElementById('removeAllBtn');
        
        if (loading) loading.style.display = 'none';
        if (extractBtn) extractBtn.style.display = '';
        if (removeAllBtn) removeAllBtn.disabled = false;
    }

    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        if (!errorMessage) return;
        
        const errorText = errorMessage.querySelector('.error-text');
        if (errorText) {
            errorText.textContent = message;
        } else {
            errorMessage.textContent = message;
        }
        errorMessage.style.display = 'block';
        errorMessage.classList.add('fade-in');
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        const errorMessage = document.getElementById('errorMessage');
        if (errorMessage) {
            errorMessage.style.display = 'none';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.resumeExtractor = new ResumeExtractor();
});
