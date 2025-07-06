document.addEventListener('DOMContentLoaded', async () => {
    const fitContent = document.getElementById('candidateFitContent');
    const errorMessage = document.getElementById('errorMessage');
    
    // Comparison functionality variables
    let compareMode = false;
    let selectedCandidates = [];
    
    function showError(msg) {
        if (!errorMessage) return;
        const errorText = errorMessage.querySelector('.error-text');
        if (errorText) errorText.textContent = msg;
        errorMessage.style.display = 'block';
    }
    
    function hideError() {
        if (errorMessage) errorMessage.style.display = 'none';
    }

    hideError();

    // Get data from localStorage (Note: This should be replaced with in-memory storage in production)
    const extractedData = JSON.parse(localStorage.getItem('extractedData') || 'null');
    const jobDescriptionData = localStorage.getItem('jobDescriptionData');
    const fitOptions = JSON.parse(localStorage.getItem('fitOptions') || '{}');

    if (!extractedData || !jobDescriptionData) {
        showError('Missing extracted data or job description. Please go back and extract first.');
        return;
    }

    // Fetch candidate fit analysis
    try {
        fitContent.innerHTML = '<div class="loading"><div class="spinner"></div><p>Analyzing candidate fit...</p></div>';
        const resumes = extractedData.extracted_data || [];
        const response = await fetch('http://localhost:8000/candidate-fit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                resume_data: resumes,
                job_description_data: jobDescriptionData,
                fit_options: fitOptions
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
            await new Promise(res => setTimeout(res, 3000));
            const pollResp = await fetch(`http://localhost:8000/candidate-fit/${jobId}`);
            const pollData = await pollResp.json();
            if (pollData.status === 'completed') {
                fitData = pollData;
                break;
            }
        }

        // Render sorted fit results with scroll and comparison functionality
        let fitResults = fitData && (fitData.fit_results || (fitData.fit_result && [fitData.fit_result]));
        if (fitResults && fitResults.length > 0) {
            fitResults = fitResults.slice().sort((a, b) => {
                const aScore = typeof a.fit_percentage === 'string' ? parseFloat(a.fit_percentage) : a.fit_percentage || 0;
                const bScore = typeof b.fit_percentage === 'string' ? parseFloat(b.fit_percentage) : b.fit_percentage || 0;
                return bScore - aScore;
            });
            const minPercent = parseFloat(fitOptions.min_fit_percent) || 0;

            let html = `<div class="fit-list fit-list-scroll">`;
            fitResults.forEach((fitResult, idx) => {
                const percent = parseFloat(fitResult.fit_percentage) || 0;
                if (percent < minPercent) return; // Skip if below threshold

                const candidateName = fitResult.candidate_name || `Candidate ${idx + 1}`;
                let fitPercent = fitResult.fit_percentage;
                if (fitPercent === undefined || fitPercent === null || fitPercent === "") {
                    fitPercent = "N/A";
                } else {
                    fitPercent = `${fitPercent}%`;
                }
                const summary = fitResult.summary || "<em>No summary provided.</em>";
                html += `
                    <div class="fit-item" style="margin-bottom:18px;">
                        <div class="fit-header" style="display:flex;align-items:center;cursor:pointer;gap:18px;padding:12px 0;" data-fit-index="${idx}">
                            <input type="checkbox" class="fit-checkbox" data-fit-index="${idx}" style="display:none;margin-right:10px;">
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
            
            // Setup comparison functionality
            setupComparisonFeature(fitResults);
            
            // Setup download functionality
            const downloadBtn = document.getElementById('downloadBtn');
            if (downloadBtn) {
                downloadBtn.onclick = async function() {
                    if (!fitResults || fitResults.length === 0) {
                        alert('No candidate fit data to download.');
                        return;
                    }
                    // Filter by minimum fit percentage
                    const minPercent = parseFloat(fitOptions.min_fit_percent) || 0;
                    const filteredResults = fitResults.filter(fitResult => {
                        const percent = parseFloat(fitResult.fit_percentage) || 0;
                        return percent >= minPercent;
                    });
                    if (filteredResults.length === 0) {
                        alert('No candidates meet the minimum fit percentage.');
                        return;
                    }
                    try {
                        // Send filteredResults to backend for Excel generation
                        const response = await fetch('http://localhost:8000/download-fit-excel', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ fit_results: filteredResults })
                        });
                        if (!response.ok) throw new Error('Failed to download Excel file.');
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'candidate_fit_results.xlsx';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                    } catch (err) {
                        alert('Error downloading Excel: ' + err.message);
                    }
                };
            }
            
            // Add expand/collapse logic
            fitContent.querySelectorAll('.fit-header').forEach(header => {
                header.addEventListener('click', function (e) {
                    // Don't expand/collapse when clicking checkbox
                    if (e.target.classList.contains('fit-checkbox')) {
                        return;
                    }
                    
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
    } catch (error) {
        showError('Failed to fetch candidate fit summary: ' + error.message);
    }
    
    // Comparison functionality setup
    function setupComparisonFeature(fitResults) {
        const compareBtn = document.getElementById('compareCandidatesBtn');
        if (compareBtn) {
            compareBtn.onclick = function() {
                compareMode = !compareMode;
                selectedCandidates = [];
                
                // Toggle compare button text
                this.textContent = compareMode ? 'Exit Compare Mode' : 'Compare Candidates';
                
                // Show/hide checkboxes
                showComparePanel(compareMode);
            };
        }
        
        // Setup checkbox event listeners
        fitContent.querySelectorAll('.fit-checkbox').forEach(cb => {
            cb.addEventListener('change', function() {
                const idx = parseInt(this.getAttribute('data-fit-index'));
                if (this.checked) {
                    if (selectedCandidates.length >= 2) {
                        this.checked = false;
                        alert('You can only compare two candidates at a time.');
                        return;
                    }
                    selectedCandidates.push(idx);
                } else {
                    selectedCandidates = selectedCandidates.filter(i => i !== idx);
                }
                
                // Show comparison if two selected
                if (selectedCandidates.length === 2) {
                    showCandidateComparison(selectedCandidates.map(i => fitResults[i]));
                } else {
                    hideCandidateComparison();
                }
            });
        });
    }
    
    // Show/hide checkboxes based on compareMode
    function showComparePanel(show) {
        fitContent.querySelectorAll('.fit-checkbox').forEach(cb => {
            cb.style.display = show ? 'inline-block' : 'none';
            cb.checked = false;
        });
        selectedCandidates = [];
        hideCandidateComparison();
    }
    
    // Comparison modal/section
    function showCandidateComparison([cand1, cand2]) {
        let modal = document.getElementById('compareModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'compareModal';
            modal.style.cssText = `
                position: fixed; left: 0; top: 0; width: 100vw; height: 100vh;
                background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 9999;
            `;
            modal.innerHTML = `
                <div style="
                    background: linear-gradient(135deg, #f1f1f3 0%, #56d6d6);
                    border: 1.5px solid rgba(255,255,255,0.35);
                    border-radius: 18px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.18);
                    padding: 32px 36px;
                    max-width: 900px;
                    max-height: 90vh;
                    overflow: auto;
                    backdrop-filter: blur(18px);
                    -webkit-backdrop-filter: blur(18px);
                    ">
                    <h2 style="margin-bottom:18px;">Candidate Comparison</h2>
                    <div id="compareModalContent" style="display:flex;gap:32px;"></div>
                    <button id="closeCompareModal" class="frosted-btn" style="margin-top: 18px;">Close</button>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Close modal when clicking outside
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            });
            
            modal.querySelector('#closeCompareModal').onclick = () => { 
                modal.style.display = 'none'; 
            };
        }
        
        const content = modal.querySelector('#compareModalContent');
        content.innerHTML = `
            <div style="flex:1; border-right: 1px solid #eee; padding-right: 16px;">
                <h3 style="color: #007bff; margin-bottom: 12px;">${cand1.candidate_name || 'Candidate 1'}</h3>
                <div style="margin-bottom: 16px;">
                    <strong>Fit Score:</strong> 
                    <span style="background: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">
                        ${cand1.fit_percentage}%
                    </span>
                </div>
                <div style="margin-bottom: 16px;">
                    <strong>Summary:</strong>
                    <p style="margin-top: 8px; line-height: 1.5;">${cand1.summary || 'No summary available'}</p>
                </div>
                ${cand1.key_matches && cand1.key_matches.length > 0 ? `
                    <div style="margin-bottom: 16px;">
                        <strong style="color: #28a745;">✓ Key Matches:</strong>
                        <ul style="margin-top: 8px; padding-left: 20px;">
                            ${cand1.key_matches.map(m => `<li style="margin-bottom: 4px;">${m}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                ${cand1.key_gaps && cand1.key_gaps.length > 0 ? `
                    <div>
                        <strong style="color: #dc3545;">⚠ Key Gaps:</strong>
                        <ul style="margin-top: 8px; padding-left: 20px;">
                            ${cand1.key_gaps.map(g => `<li style="margin-bottom: 4px;">${g}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
            <div style="flex:1; padding-left: 16px;">
                <h3 style="color: #007bff; margin-bottom: 12px;">${cand2.candidate_name || 'Candidate 2'}</h3>
                <div style="margin-bottom: 16px;">
                    <strong>Fit Score:</strong> 
                    <span style="background: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">
                        ${cand2.fit_percentage}%
                    </span>
                </div>
                <div style="margin-bottom: 16px;">
                    <strong>Summary:</strong>
                    <p style="margin-top: 8px; line-height: 1.5;">${cand2.summary || 'No summary available'}</p>
                </div>
                ${cand2.key_matches && cand2.key_matches.length > 0 ? `
                    <div style="margin-bottom: 16px;">
                        <strong style="color: #28a745;">✓ Key Matches:</strong>
                        <ul style="margin-top: 8px; padding-left: 20px;">
                            ${cand2.key_matches.map(m => `<li style="margin-bottom: 4px;">${m}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                ${cand2.key_gaps && cand2.key_gaps.length > 0 ? `
                    <div>
                        <strong style="color: #dc3545;">⚠ Key Gaps:</strong>
                        <ul style="margin-top: 8px; padding-left: 20px;">
                            ${cand2.key_gaps.map(g => `<li style="margin-bottom: 4px;">${g}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
        modal.style.display = 'flex';
    }
    
    function hideCandidateComparison() {
        const modal = document.getElementById('compareModal');
        if (modal) modal.style.display = 'none';
    }
});