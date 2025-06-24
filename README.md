# Resume-Info-Extractor# AI Resume Extractor

A powerful web application that uses Google's Gemini AI to extract structured data from resume files in multiple formats (PDF, DOC, DOCX, XLS, XLSX, TXT).

## Features

- **Multiple File Format Support**: PDF, DOC, DOCX, XLS, XLSX, TXT
- **Drag & Drop Interface**: Modern, intuitive file upload experience
- **AI-Powered Extraction**: Uses Google Gemini AI for accurate data extraction
- **Multiple Export Formats**: JSON, CSV, Excel, PDF reports
- **Batch Processing**: Process multiple resumes simultaneously
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
resume-extractor/
├── index.html              # Frontend HTML
├── style.css              # CSS styling
├── scripts.js             # Frontend JavaScript
├── main.py                # FastAPI backend server
├── resume_extractor.py    # AI extraction logic
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd resume-extractor
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 4. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and paste it in your `.env` file

### 5. Start the Backend Server

```bash
python main.py
```

The FastAPI server will start on `http://localhost:8000`

### 6. Open the Frontend

Open `index.html` in your web browser or serve it using a local server:

```bash
# Using Python's built-in server
python -m http.server 3000

# Then open http://localhost:3000 in your browser
```

## API Endpoints

### Extract Resume Data
- **POST** `/extract-resume`
- **Description**: Upload resume files and extract structured data
- **Parameters**: `files` (multipart form data)
- **Response**: JSON with extracted data

### Download Data
- **POST** `/download-data`
- **Description**: Convert extracted data to specified format
- **Body**: JSON with `data` and `format` fields
- **Response**: File download in requested format

## Usage

1. **Upload Files**: Drag and drop resume files or click to browse
2. **Extract Data**: Click "Extract Data" to process files with AI
3. **Review Results**: View extracted data in the preview panel
4. **Download**: Choose your preferred format and download the results

## Supported File Formats

- **PDF**: Portable Document Format
- **DOC/DOCX**: Microsoft Word documents
- **XLS/XLSX**: Microsoft Excel spreadsheets
- **TXT**: Plain text files

## Extracted Data Structure

The AI extracts the following information:

- **Personal Information**: Name, email, phone, location, LinkedIn, portfolio
- **Professional Summary**: Career objective or summary
- **Skills**: Technical and soft skills
- **Work Experience**: Job titles, companies, durations, descriptions
- **Education**: Degrees, institutions, graduation years
- **Certifications**: Professional certifications and licenses
- **Projects**: Personal or professional projects
- **Languages**: Language proficiency
- **Awards**: Recognition and achievements

## Export Formats

- **JSON**: Structured data format
- **CSV**: Spreadsheet-compatible format
- **Excel**: Multi-sheet Excel workbook
- **PDF**: Formatted report

## Technical Details

### Frontend Technologies
- HTML5 with semantic markup
- CSS3 with modern styling (gradients, animations, flexbox)
- Vanilla JavaScript (ES6+)
- Responsive design principles

### Backend Technologies
- FastAPI (Python web framework)
- Google Generative AI (Gemini)
- File processing libraries (PyPDF2, python-docx, pandas)
- Report generation (ReportLab)

### AI Integration
- Uses Google's Gemini 1.5 Flash model
- Custom prompt engineering for resume parsing
- JSON-structured data extraction
- Error handling and data validation

## Error Handling

The application includes comprehensive error handling for:
- Unsupported file formats
- Corrupted files
- AI API errors
- Network connectivity issues
- Invalid data formats

## Security Considerations

- File type validation
- File size limits
- Temporary file cleanup
- CORS protection
- API key security

## Performance Optimization

- Asynchronous file processing
- Efficient text extraction
- Optimized AI prompts
- Client-side file validation
- Progressive loading states

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY not found"**
   - Ensure your `.env` file exists and contains the API key
   - Verify the API key is valid

2. **"Unsupported file type"**
   - Check that your file is in a supported format
   - Ensure file extensions are correct

3. **"Server connection failed"**
   - Verify the FastAPI server is running on port 8000
   - Check for CORS issues if using a different port

4. **"AI extraction failed"**
   - Check your Gemini API quota
   - Verify the file contains readable text

### Debug Mode

Enable debug mode in the `.env` file:
```
DEBUG=True
```

This will provide detailed error messages and logging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error logs
3. Open an issue on GitHub
4. Provide detailed error information and steps to reproduce

## Future Enhancements

- Support for more file formats
- Batch processing improvements
- Advanced data validation
- Custom extraction templates
- Integration with HR systems
- Resume scoring and ranking