document.addEventListener('DOMContentLoaded', () => {
    // Debug: Log when script starts
    console.log('Job description script loaded');
    
    // Retrieve extractedData from localStorage
    const extractedData = JSON.parse(localStorage.getItem('extractedData') || 'null');
    if (!extractedData) {
        // Optionally redirect or show error if not present
        console.warn('No extracted data found in localStorage');
        alert('No extracted data found. Please extract resumes first.');
        window.location.href = 'index.html';
        return;
    }
    
    console.log('Extracted data found:', extractedData);
    
    // Home button logic (if present)
    const homeBtn = document.getElementById('homeBtn');
    if (homeBtn) {
        homeBtn.onclick = function() {
            console.log('Home button clicked');
            window.location.href = 'index.html';
        };
    }

    // Get Best Candidate button logic
    const getBestFitBtn = document.getElementById('getBestFitBtn');
    if (getBestFitBtn) {
        console.log('Get Best Fit button found, adding event listener');
        
        getBestFitBtn.onclick = function() {
            console.log('Get Best Fit button clicked');
            
            try {
                // Collect values from the fields
                const jdInput = document.getElementById('jdInput')?.value || '';
                const priorityKeywords = document.getElementById('priorityKeywords')?.value || '';
                const minExperience = document.getElementById('minExperience')?.value || '';
                const eduRequirements = document.getElementById('eduRequirements')?.value || '';
                const weightSkills = document.getElementById('weightSkills')?.value || '40';
                const weightExperience = document.getElementById('weightExperience')?.value || '40';
                const weightEducation = document.getElementById('weightEducation')?.value || '20';
                const minFitPercent = document.getElementById('minFitPercent')?.value || '0';
                const cfRequirements = document.getElementById('cfRequirements')?.value || '';

                console.log('Form data collected:', {
                    jdInput: jdInput.length + ' characters',
                    priorityKeywords,
                    minExperience,
                    eduRequirements,
                    weightSkills,
                    weightExperience,
                    weightEducation,
                    minFitPercent,
                    cfRequirements
                });

                // Validate required fields
                if (!jdInput.trim()) {
                    alert('Please enter a job description');
                    return;
                }

                // Validate weights sum to 100
                const totalWeight = parseInt(weightSkills) + parseInt(weightExperience) + parseInt(weightEducation);
                if (totalWeight !== 100) {
                    alert(`Total weight should be 100, currently it's ${totalWeight}. Please adjust the weights.`);
                    return;
                }

                // Save to localStorage
                localStorage.setItem('jobDescriptionData', jdInput);
                
                const fitOptions = {
                    priority_keywords: priorityKeywords,
                    min_experience: minExperience,
                    edu_requirements: eduRequirements,
                    weight_skills: weightSkills,
                    weight_experience: weightExperience,
                    weight_education: weightEducation,
                    min_fit_percent: minFitPercent,
                    cf_requirements: cfRequirements
                };
                
                localStorage.setItem('fitOptions', JSON.stringify(fitOptions));
                
                console.log('Data saved to localStorage:', {
                    jobDescriptionData: 'saved',
                    fitOptions: fitOptions
                });

                // Check if candidate_fit.html exists before navigating
                console.log('Attempting to navigate to candidate_fit.html');
                
                // For debugging - you can comment this out once it works
                // alert('About to navigate to candidate_fit.html. Check console for details.');
                
                window.location.href = 'candidate_fit.html';
                
            } catch (error) {
                console.error('Error in getBestFitBtn click handler:', error);
                alert('An error occurred: ' + error.message);
            }
        };
    } else {
        console.error('Get Best Fit button not found in DOM');
    }

    // Optional: Drag & drop for job description area
    const jdUploadArea = document.getElementById('jdUploadArea');
    const jdInput = document.getElementById('jdInput');
    
    if (jdUploadArea && jdInput) {
        console.log('Setting up drag and drop functionality');
        
        jdUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            jdUploadArea.classList.add('dragover');
        });
        
        jdUploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            jdUploadArea.classList.remove('dragover');
        });
        
        jdUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            jdUploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.items) {
                for (let i = 0; i < e.dataTransfer.items.length; i++) {
                    const item = e.dataTransfer.items[i];
                    if (item.kind === 'string' && item.type === 'text/plain') {
                        item.getAsString((str) => {
                            jdInput.value = str;
                            console.log('Text dropped into job description area');
                        });
                        return;
                    } else if (item.kind === 'file') {
                        const file = item.getAsFile();
                        if (file && (file.type.startsWith('text/') || file.name.toLowerCase().endsWith('.txt'))) {
                            const reader = new FileReader();
                            reader.onload = (evt) => {
                                jdInput.value = evt.target.result;
                                console.log('File content loaded into job description area');
                            };
                            reader.readAsText(file);
                            return;
                        }
                    }
                }
            }
            
            const text = e.dataTransfer.getData('text/plain');
            if (text) {
                jdInput.value = text;
                console.log('Plain text dropped into job description area');
            }
        });
    }
    
    console.log('Job description script initialization complete');
});