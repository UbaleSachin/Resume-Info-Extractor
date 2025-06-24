class ResumeExtractor {
    constructor() {
        this.selectedFiles = [];
        this.extractedData = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const extractBtn = document.getElementById('extractBtn');
        const downloadBtn = document.getElementById('downloadBtn');

        // File upload events
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Button events
        extractBtn.addEventListener('click', () => this.extractData());
        downloadBtn.addEventListener('click', () => this.downloadData());
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadArea').classList.remove('dragover');
        
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
                <button class="remove-file" onclick="resumeExtractor.removeFile(${index})">×</button>
            `;
            
            filesList.appendChild(fileItem);
        });
    }

    removeFile(index) {
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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.extractedData = data;
            this.displayResults(data);
            
        } catch (error) {
            console.error('Error extracting data:', error);
            this.showError('Failed to extract data. Please make sure the server is running and try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const dataPreview = document.getElementById('dataPreview');
        
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Display formatted JSON data
        dataPreview.textContent = JSON.stringify(data, null, 2);
    }

    async downloadData() {
        if (!this.extractedData) {
            this.showError('No data to download. Please extract data first.');
            return;
        }

        const format = document.getElementById('downloadFormat').value;
        
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

    showLoading() {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('filesPreview').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('filesPreview').style.display = 'block';
    }

    showError(message) {
        const errorMessage = document.getElementById('errorMessage');
        const errorText = errorMessage.querySelector('.error-text');
        errorText.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.classList.add('fade-in');
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        document.getElementById('errorMessage').style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.resumeExtractor = new ResumeExtractor();
});