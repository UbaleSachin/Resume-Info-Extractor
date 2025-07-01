document.addEventListener('DOMContentLoaded', () => {
    // Home button logic (if present)
    const homeBtn = document.getElementById('homeBtn');
    if (homeBtn) {
        homeBtn.onclick = function() {
            localStorage.removeItem('extractedData');
            localStorage.removeItem('jobDescriptionData');
            localStorage.removeItem('fitOptions');
            window.location.href = 'index.html';
        };
    }

    // Get Best Candidate button logic
    const getBestFitBtn = document.getElementById('getBestFitBtn');
    if (getBestFitBtn) {
        getBestFitBtn.onclick = function() {
            // Check if extractedData exists
            if (!localStorage.getItem('extractedData')) {
                alert('Please upload and extract resumes first.');
                window.location.href = 'index.html';
                return;
            }
            // Collect values from the fields
            const jdInput = document.getElementById('jdInput')?.value || '';
            const priorityKeywords = document.getElementById('priorityKeywords')?.value || '';
            const minExperience = document.getElementById('minExperience')?.value || '';
            const eduRequirements = document.getElementById('eduRequirements')?.value || '';
            const weightSkills = document.getElementById('weightSkills')?.value || '';
            const weightExperience = document.getElementById('weightExperience')?.value || '';
            const weightEducation = document.getElementById('weightEducation')?.value || '';
            const minFitPercent = document.getElementById('minFitPercent')?.value || '';

            // Save to localStorage
            localStorage.setItem('jobDescriptionData', jdInput);
            const fitOptions = {
                priority_keywords: priorityKeywords,
                min_experience: minExperience,
                edu_requirements: eduRequirements,
                weight_skills: weightSkills,
                weight_experience: weightExperience,
                weight_education: weightEducation,
                min_fit_percent: minFitPercent
            };
            localStorage.setItem('fitOptions', JSON.stringify(fitOptions));
            window.location.href = 'candidate_fit.html';
        };
    }

    // Optional: Drag & drop for job description area
    const jdUploadArea = document.getElementById('jdUploadArea');
    const jdInput = document.getElementById('jdInput');
    if (jdUploadArea && jdInput) {
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
                        });
                        return;
                    } else if (item.kind === 'file') {
                        const file = item.getAsFile();
                        if (file && (file.type.startsWith('text/') || file.name.toLowerCase().endsWith('.txt'))) {
                            const reader = new FileReader();
                            reader.onload = (evt) => {
                                jdInput.value = evt.target.result;
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
            }
        });
    }
});