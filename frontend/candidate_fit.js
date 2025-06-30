document.addEventListener('DOMContentLoaded', async () => {
    const fitContent = document.getElementById('candidateFitContent');
    const errorMessage = document.getElementById('errorMessage');
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

    // Get data from localStorage
    const extractedData = JSON.parse(localStorage.getItem('extractedData') || 'null');
    const jobDescriptionData = localStorage.getItem('jobDescriptionData');

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

        // Render sorted fit results with scroll
        let fitResults = fitData && (fitData.fit_results || (fitData.fit_result && [fitData.fit_result]));
        if (fitResults && fitResults.length > 0) {
            fitResults = fitResults.slice().sort((a, b) => {
                const aScore = typeof a.fit_percentage === 'string' ? parseFloat(a.fit_percentage) : a.fit_percentage || 0;
                const bScore = typeof b.fit_percentage === 'string' ? parseFloat(b.fit_percentage) : b.fit_percentage || 0;
                return bScore - aScore;
            });

            let html = `<div class="fit-list fit-list-scroll">`;
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
    } catch (error) {
        showError('Failed to fetch candidate fit summary: ' + error.message);
    }
});