// Event Handlers
fileInput.addEventListener('change', function() {
    if (this.files && this.files[0]) {
        cleanGlobalBtn.disabled = false;
        extractBtn.disabled = false;
        cleanPiecesBtn.disabled = true;
        cornersBtn.disabled = true;
        matchingBtn.disabled = true;
        showStatus('File selected: ' + this.files[0].name, 'info');
        uploadedFilename = null;
        selectedVariantBlob = null;
        globalCleanResults.classList.remove('active');
        extractionResults.classList.remove('active');
        pieceCleanResults.classList.remove('active');
        edgeResults.classList.remove('active');
        matchResults.classList.remove('active');
        generatedVariants.classList.remove('active');
    }
});

generateBtn.addEventListener('click', async function() {
    // Disable buttons
    generateBtn.disabled = true;
    cleanGlobalBtn.disabled = true;
    extractBtn.disabled = true;
    cleanPiecesBtn.disabled = true;
    cornersBtn.disabled = true;
    matchingBtn.disabled = true;

    // Read seed from input
    const seedInput = document.getElementById('seedInput');
    const seedValue = seedInput.value ? parseInt(seedInput.value) : null;

    // If seed is provided, generate only 1 variant, otherwise generate 3
    const numVariants = seedValue !== null ? 1 : 3;

    const statusMsg = seedValue !== null
        ? `Generiere Puzzle mit Seed ${seedValue}...`
        : 'Generiere 3 Puzzle-Varianten...';
    showStatus(statusMsg, 'info');

    loadingText.textContent = 'Generiere Puzzles (kann 5-15 Sekunden dauern)...';
    loading.classList.add('active');
    generatedVariants.classList.remove('active');
    globalCleanResults.classList.remove('active');
    extractionResults.classList.remove('active');
    pieceCleanResults.classList.remove('active');
    edgeResults.classList.remove('active');
    matchResults.classList.remove('active');

    try {
        // Generate variants
        const variants = [];
        for (let i = 0; i < numVariants; i++) {
            loadingText.textContent = `Generiere Puzzle ${i + 1}/${numVariants}...`;

            // Build request body
            const requestBody = {
                layout: '2x3'
                // cut_types will be randomly selected on server-side
            };

            // Add seed if provided
            if (seedValue !== null) {
                requestBody.seed = seedValue;
            }

            const response = await fetch('/puzzle-gen/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Generation failed');
            }

            variants.push(result);
        }

        // Display variants
        displayVariants(variants);

        const successMsg = seedValue !== null
            ? `Puzzle mit Seed ${seedValue} erfolgreich generiert!`
            : '3 Puzzle-Varianten erfolgreich generiert! Wähle eine aus.';
        showStatus(successMsg, 'success');

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        generateBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

extractBtn.addEventListener('click', async function() {
    const file = fileInput.files[0];
    if (!file && !selectedVariantBlob) {
        showStatus('Please select a file or generate a puzzle first', 'error');
        return;
    }

    // Disable extract button
    extractBtn.disabled = true;
    cornersBtn.disabled = true;
    matchingBtn.disabled = true;

    // Upload file or use selected variant
    showStatus('Uploading file...', 'info');
    loadingText.textContent = 'Uploading file...';
    loading.classList.add('active');
    extractionResults.classList.remove('active');
    edgeResults.classList.remove('active');
    matchResults.classList.remove('active');

    const formData = new FormData();
    if (selectedVariantBlob) {
        // Generate filename with seed and timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const filename = `puzzle_seed${selectedVariantSeed}_${timestamp}.png`;
        formData.append('file', selectedVariantBlob, filename);
    } else {
        // Use uploaded file
        formData.append('file', file);
    }

    try {
        const uploadResponse = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const uploadResult = await uploadResponse.json();

        if (!uploadResult.success) {
            throw new Error(uploadResult.error || 'Upload failed');
        }

        uploadedFilename = uploadResult.filename;
        showStatus('File uploaded. Extracting pieces...', 'info');
        loadingText.textContent = 'Extracting puzzle pieces...';

        // Extract pieces
        const extractResponse = await fetch(`/extract/${uploadedFilename}`);
        const extractResult = await extractResponse.json();

        if (!extractResult.success) {
            throw new Error(extractResult.error || 'Extraction failed');
        }

        // Display extraction results
        displayExtractionResults(extractResult);
        showStatus(`Successfully extracted ${extractResult.num_pieces} puzzle pieces!`, 'success');

        // Enable piece cleaning and edge detection buttons
        cleanPiecesBtn.disabled = false;
        cornersBtn.disabled = false;

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        extractBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

cleanGlobalBtn.addEventListener('click', async function() {
    const file = fileInput.files[0];
    if (!file && !selectedVariantBlob) {
        showStatus('Please select a file or generate a puzzle first', 'error');
        return;
    }

    // First upload the file if not already uploaded
    if (!uploadedFilename) {
        cleanGlobalBtn.disabled = true;
        extractBtn.disabled = true;

        showStatus('Uploading file...', 'info');
        loadingText.textContent = 'Uploading file...';
        loading.classList.add('active');

        const formData = new FormData();
        if (selectedVariantBlob) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const filename = `puzzle_seed${selectedVariantSeed}_${timestamp}.png`;
            formData.append('file', selectedVariantBlob, filename);
        } else {
            formData.append('file', file);
        }

        try {
            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const uploadResult = await uploadResponse.json();

            if (!uploadResult.success) {
                throw new Error(uploadResult.error || 'Upload failed');
            }

            uploadedFilename = uploadResult.filename;
        } catch (error) {
            showStatus('Error: ' + error.message, 'error');
            cleanGlobalBtn.disabled = false;
            extractBtn.disabled = false;
            loading.classList.remove('active');
            return;
        }
    }

    // Now apply global cleaning
    cleanGlobalBtn.disabled = true;
    extractBtn.disabled = true;

    showStatus('Applying global cleaning...', 'info');
    loadingText.textContent = 'Cleaning image globally...';
    loading.classList.add('active');

    try {
        const cleanResponse = await fetch(`/clean-global/${uploadedFilename}`);
        const cleanResult = await cleanResponse.json();

        if (!cleanResult.success && !cleanResult.warning) {
            throw new Error(cleanResult.error || 'Global cleaning failed');
        }

        // Display results
        displayGlobalCleanResults(cleanResult);

        if (cleanResult.warning) {
            showStatus(cleanResult.warning + ' ' + cleanResult.message, 'warning');
        } else {
            showStatus('Global cleaning completed!', 'success');
        }

        // Enable extraction button
        extractBtn.disabled = false;

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        cleanGlobalBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

cleanPiecesBtn.addEventListener('click', async function() {
    if (!uploadedFilename) {
        showStatus('Please extract pieces first', 'error');
        return;
    }

    cleanPiecesBtn.disabled = true;
    cornersBtn.disabled = true;

    showStatus('Cleaning puzzle pieces...', 'info');
    loadingText.textContent = 'Cleaning pieces...';
    loading.classList.add('active');

    try {
        const cleanResponse = await fetch(`/clean-pieces/${uploadedFilename}`);
        const cleanResult = await cleanResponse.json();

        if (!cleanResult.success) {
            throw new Error(cleanResult.error || 'Piece cleaning failed');
        }

        // Display results
        displayPieceCleanResults(cleanResult);
        showStatus(`Successfully cleaned ${cleanResult.num_pieces} pieces!`, 'success');

        // Enable edge detection button
        cornersBtn.disabled = false;

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        cleanPiecesBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

cornersBtn.addEventListener('click', async function() {
    if (!uploadedFilename) {
        showStatus('Please extract pieces first', 'error');
        return;
    }

    // Disable button
    cornersBtn.disabled = true;
    matchingBtn.disabled = true;

    showStatus('Detecting edges...', 'info');
    loadingText.textContent = 'Detecting edges...';
    loading.classList.add('active');

    try {
        const edgeResponse = await fetch(`/detect-edges/${uploadedFilename}`);
        const edgeResult = await edgeResponse.json();

        if (!edgeResult.success) {
            throw new Error(edgeResult.error || 'Edge detection failed');
        }

        // Display edge detection results
        displayEdgeResults(edgeResult);
        showStatus(`Successfully detected ${edgeResult.num_edges} edges!`, 'success');

        // Enable matching button
        matchingBtn.disabled = false;

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        cornersBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

matchingBtn.addEventListener('click', async function() {
    if (!uploadedFilename) {
        showStatus('Please detect edges first', 'error');
        return;
    }

    // Disable button
    matchingBtn.disabled = true;

    showStatus('Finding matches...', 'info');
    loadingText.textContent = 'Finding edge matches...';
    loading.classList.add('active');

    try {
        const matchResponse = await fetch(`/match-edges/${uploadedFilename}?min_score=0.0`);
        const matchResult = await matchResponse.json();

        if (!matchResult.success) {
            throw new Error(matchResult.error || 'Matching failed');
        }

        // Display match results
        displayMatchResults(matchResult);
        showStatus(`Successfully found ${matchResult.num_matches} matches!`, 'success');
        solverBtn.disabled = false;

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        matchingBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

solverBtn.addEventListener('click', async function() {
    if (!uploadedFilename) {
        showStatus('Please extract pieces first', 'error');
        return;
    }

    // Disable button
    solverBtn.disabled = true;

    showStatus('Solving puzzle...', 'info');
    loadingText.textContent = 'Solving puzzle...';
    loading.classList.add('active');
    solutionResults.classList.remove('active');

    try {
        const solveResponse = await fetch(`/solve-puzzle/${uploadedFilename}`);
        const solveResult = await solveResponse.json();

        if (!solveResult.success) {
            throw new Error(solveResult.error || 'Puzzle solving failed');
        }

        // Display solution results
        displaySolutionResults(solveResult);
        showStatus(`Puzzle solved! ${solveResult.solution.pieces_placed}/${solveResult.solution.total_pieces} pieces placed (${(solveResult.solution.confidence * 100).toFixed(1)}% confidence)`, 'success');

    } catch (error) {
        showStatus('Error: ' + error.message, 'error');
        solverBtn.disabled = false;
    } finally {
        loading.classList.remove('active');
    }
});

resetBtn.addEventListener('click', function() {
    // Reset all state variables
    uploadedFilename = null;
    selectedVariantBlob = null;
    selectedVariantSeed = null;

    // Reset file input
    fileInput.value = '';

    // Reset button states
    generateBtn.disabled = false;
    cleanGlobalBtn.disabled = true;
    extractBtn.disabled = true;
    cleanPiecesBtn.disabled = true;
    cornersBtn.disabled = true;
    matchingBtn.disabled = true;
    solverBtn.disabled = true;

    // Hide all result sections
    generatedVariants.classList.remove('active');
    globalCleanResults.classList.remove('active');
    extractionResults.classList.remove('active');
    pieceCleanResults.classList.remove('active');
    edgeResults.classList.remove('active');
    matchResults.classList.remove('active');
    solutionResults.classList.remove('active');

    // Hide loading
    loading.classList.remove('active');

    // Reset status message
    status.textContent = '';
    status.className = 'status';

    // Show reset message
    showStatus('Zurückgesetzt. Wähle ein Bild oder generiere ein Puzzle.', 'info');
});

// Display Functions
function displayExtractionResults(data) {
    // Show results section
    extractionResults.classList.add('active');

    // Display statistics
    const statsGrid = document.getElementById('extractionStatsGrid');
    const stats = data.statistics;
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="value">${data.num_pieces}</div>
            <div class="label">Pieces Found</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.avg_area)}</div>
            <div class="label">Avg Area (px²)</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.min_area)}</div>
            <div class="label">Min Area (px²)</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.max_area)}</div>
            <div class="label">Max Area (px²)</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.avg_perimeter)}</div>
            <div class="label">Avg Perimeter (px)</div>
        </div>
    `;

    // Display individual pieces
    const piecesGrid = document.getElementById('extractedPiecesGrid');
    piecesGrid.innerHTML = '';

    data.images.pieces.forEach((pieceFilename, idx) => {
        const piece = data.pieces[idx];
        const pieceCard = document.createElement('div');
        pieceCard.className = 'piece-card';
        pieceCard.innerHTML = `
            <img src="/output/${pieceFilename}" alt="Piece ${idx}">
            <div class="piece-info">
                <div><strong>Piece ID:</strong> ${idx}</div>
                <div><strong>Area:</strong> ${Math.round(piece.area)} px²</div>
                <div><strong>Perimeter:</strong> ${Math.round(piece.perimeter)} px</div>
                <div><strong>Center:</strong> (${piece.center[0]}, ${piece.center[1]})</div>
                <div><strong>Size:</strong> ${piece.bbox.width} × ${piece.bbox.height}</div>
            </div>
        `;
        piecesGrid.appendChild(pieceCard);
    });

    // Scroll to results
    extractionResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayEdgeResults(data) {
    // Show edge results section
    edgeResults.classList.add('active');

    // Display edge statistics
    const edgeStatsGrid = document.getElementById('edgeStatsGrid');
    const stats = data.statistics;
    edgeStatsGrid.innerHTML = `
        <div class="stat-card">
            <div class="value">${data.num_edges}</div>
            <div class="label">Total Edges</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.avg_edge_length)}</div>
            <div class="label">Avg Length (px)</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.flat_edges}</div>
            <div class="label">Flat Edges</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.tab_edges}</div>
            <div class="label">Tab Edges</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.slot_edges}</div>
            <div class="label">Slot Edges</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.min_edge_length)} - ${Math.round(stats.max_edge_length)}</div>
            <div class="label">Length Range (px)</div>
        </div>
    `;

    // Display pieces with edges
    const edgePiecesGrid = document.getElementById('edgePiecesGrid');
    edgePiecesGrid.innerHTML = '';

    data.images.edge_pieces.forEach((edgeFilename, idx) => {
        const edgeInfo = data.edge_info[idx];
        const pieceCard = document.createElement('div');
        pieceCard.className = 'piece-card';

        // Build edge info HTML
        let edgeInfoHTML = '';
        for (const [edgeType, info] of Object.entries(edgeInfo.edges)) {
            edgeInfoHTML += `<div><strong>${edgeType}:</strong> ${info.classification} (${Math.round(info.length)}px)</div>`;
        }

        pieceCard.innerHTML = `
            <img src="/output/${edgeFilename}" alt="Piece ${idx} edges">
            <div class="piece-info">
                <div><strong>Piece ID:</strong> ${idx}</div>
                ${edgeInfoHTML}
            </div>
        `;
        edgePiecesGrid.appendChild(pieceCard);
    });

    // Scroll to edge results
    edgeResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

 function displayMatchResults(data) {
    // Show match results section
    matchResults.classList.add('active');

    // Display match statistics
    const matchStatsGrid = document.getElementById('matchStatsGrid');
    const stats = data.statistics;
    matchStatsGrid.innerHTML = `
        <div class="stat-card">
            <div class="value">${data.num_matches}</div>
            <div class="label">Total Matches</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.avg_score.toFixed(3)}</div>
            <div class="label">Avg Score</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.high_confidence_matches}</div>
            <div class="label">High Confidence (≥0.8)</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.medium_confidence_matches}</div>
            <div class="label">Medium Confidence (≥0.6)</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.tab_slot_matches}</div>
            <div class="label">Tab-Slot Matches</div>
        </div>
    `;

    // Display piece classification
    const pieceClass = stats.piece_classification;
    const classGrid = document.getElementById('pieceClassificationGrid');
    classGrid.innerHTML = `
        <div class="stat-card">
            <div class="value">${pieceClass.corner_pieces}</div>
            <div class="label">Corner Pieces</div>
            <div class="detail">${pieceClass.corner_piece_ids.join(', ')}</div>
        </div>
        <div class="stat-card">
            <div class="value">${pieceClass.border_pieces}</div>
            <div class="label">Border Pieces</div>
            <div class="detail">${pieceClass.border_piece_ids.join(', ')}</div>
        </div>
        <div class="stat-card">
            <div class="value">${pieceClass.interior_pieces}</div>
            <div class="label">Interior Pieces</div>
            <div class="detail">${pieceClass.interior_piece_ids.join(', ')}</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.matches_by_rotation['0°']}</div>
            <div class="label">No Rotation</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.matches_by_rotation['90°']}</div>
            <div class="label">90° Rotation</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.matches_by_rotation['180°']}</div>
            <div class="label">180° Rotation</div>
        </div>
        <div class="stat-card">
            <div class="value">${stats.matches_by_rotation['270°']}</div>
            <div class="label">270° Rotation</div>
        </div>
    `;

    // Display match visualizations with larger cards
    const matchVizGrid = document.getElementById('matchVisualizationsGrid');
    matchVizGrid.innerHTML = '';

    if (data.images && data.images.match_visualizations) {
        data.images.match_visualizations.forEach((matchFilename) => {
            const matchCard = document.createElement('div');
            matchCard.className = 'piece-card match-card-large';
            matchCard.innerHTML = `
                <img src="/output/${matchFilename}" alt="Match visualization"
                     style="cursor: pointer;"
                     onclick="window.open('/output/${matchFilename}', '_blank')">
                <div class="piece-info" style="text-align: center; padding: 10px;">
                    <small style="color: #666;">Click image to view full size</small>
                </div>
            `;
            matchVizGrid.appendChild(matchCard);
        });
    }

    // Display top matches table
    const topMatchesTable = document.getElementById('topMatchesTable');
    let tableHTML = `
        <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
            <thead>
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 10px; border: 1px solid #ddd;">Rank</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Piece 1</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Edge 1</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Piece 2</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Edge 2</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Score</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Rotation</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Length Sim</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Shape Sim</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.matches.slice(0, 20).forEach((match, idx) => {
        const scoreColor = match.scores.compatibility >= 0.8 ? '#4CAF50' :
                          match.scores.compatibility >= 0.6 ? '#FF9800' : '#f44336';
        tableHTML += `
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${idx + 1}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.edge1.piece_id}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.edge1.edge_type}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.edge2.piece_id}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.edge2.edge_type}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center; color: ${scoreColor}; font-weight: bold;">
                    ${match.scores.compatibility.toFixed(3)}
                </td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.rotation.degrees}°</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.scores.length_similarity.toFixed(2)}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">${match.scores.shape_similarity.toFixed(2)}</td>
            </tr>
        `;
    });

    tableHTML += `
            </tbody>
        </table>
    `;
    topMatchesTable.innerHTML = tableHTML;

    // Scroll to match results
    matchResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displaySolutionResults(data) {
    // Show results section
    solutionResults.classList.add('active');

    const solutionContent = document.getElementById('solutionContent');
    const solution = data.solution;

    let html = `
        <div class="info-box">
            <h4>📊 Lösungs-Statistik</h4>
            <p><strong>Gittergröße:</strong> ${solution.grid_rows} × ${solution.grid_cols}</p>
            <p><strong>Platzierte Teile:</strong> ${solution.pieces_placed} / ${solution.total_pieces}</p>
            <p><strong>Vertrauen:</strong> ${(solution.confidence * 100).toFixed(1)}%</p>
            <p><strong>Verwendete Matches:</strong> ${solution.matches_used}</p>
        </div>

        <div class="info-box">
            <h4>📍 Gitter-Layout</h4>
            <table style="border-collapse: collapse; margin: 10px auto;">
    `;

    // Create grid table
    for (let row = 0; row < solution.grid_rows; row++) {
        html += '<tr>';
        for (let col = 0; col < solution.grid_cols; col++) {
            const pieceId = solution.grid_layout[row][col];
            const cellContent = pieceId !== null ? `P${pieceId}` : '?';
            const cellClass = pieceId !== null ? 'grid-cell-filled' : 'grid-cell-empty';
            html += `<td class="${cellClass}" style="border: 1px solid #ccc; padding: 8px; text-align: center; min-width: 50px;">${cellContent}</td>`;
        }
        html += '</tr>';
    }

    html += `
            </table>
        </div>

        <div class="info-box">
            <h4>🔄 Teile-Rotationen</h4>
            <div style="max-height: 200px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f0f0f0;">
                            <th style="padding: 5px; border: 1px solid #ddd;">Piece</th>
                            <th style="padding: 5px; border: 1px solid #ddd;">Position (Row, Col)</th>
                            <th style="padding: 5px; border: 1px solid #ddd;">Rotation</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    for (const piece of solution.placed_pieces) {
        html += `
            <tr>
                <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">P${piece.piece_id}</td>
                <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">(${piece.grid_position.row}, ${piece.grid_position.col})</td>
                <td style="padding: 5px; border: 1px solid #ddd; text-align: center;">${piece.rotation.toFixed(0)}°</td>
            </tr>
        `;
    }

    html += `
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // Add visualization images
    if (data.images && data.images.solution_visualizations) {
        html += '<div class="images-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px;">';
        for (const imgFilename of data.images.solution_visualizations) {
            html += `
                <div class="image-item piece-card" style="text-align: center;">
                    <img src="/output/${imgFilename}" alt="Solution visualization" 
                         style="width: 100%; cursor: pointer;"
                         onclick="window.open('/output/${imgFilename}', '_blank')">
                    <p style="margin-top: 10px; font-weight: bold;">
                        ${imgFilename.includes('assembled') ? 'Zusammengesetztes Puzzle' : 
                          imgFilename.includes('grid') ? 'Gitter-Diagramm' : 
                          imgFilename.includes('rotations') ? 'Rotations-Guide' : 'Visualisierung'}
                    </p>
                    <p style="color: #666; font-size: 0.9em;">Klicke zum Vergrößern</p>
                </div>
            `;
        }
        html += '</div>';
    }

    solutionContent.innerHTML = html;

    // Scroll to results
    solutionResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayVariants(variants) {
    // Show variants section
    generatedVariants.classList.add('active');

    const variantsGrid = document.getElementById('variantsGrid');
    variantsGrid.innerHTML = '';

    variants.forEach((variant, idx) => {
        const variantCard = document.createElement('div');
        variantCard.className = 'piece-card';
        variantCard.style.cursor = 'pointer';
        variantCard.style.border = '3px solid transparent';
        variantCard.style.transition = 'all 0.3s';

        // Create metadata display
        const metadata = variant.metadata;
        const metadataHTML = `
            <div style="font-size: 0.9em; color: #555; margin-bottom: 10px;">
                <strong>Variante ${idx + 1}</strong><br>
                Seed: ${metadata.seed} | Layout: ${metadata.layout} | Teile: ${metadata.piece_count}<br>
                Cuts: ${metadata.cut_types.join(', ')}
            </div>
        `;

        variantCard.innerHTML = `
            ${metadataHTML}
            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 5px;">Kamera-Bild (mit Rauschen)</h4>
                <img src="data:image/png;base64,${variant.noisy_image}" alt="Noisy puzzle" style="width: 100%; border-radius: 5px;">
            </div>
            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 5px;">Lösung</h4>
                <img src="data:image/png;base64,${variant.solution_image}" alt="Solution puzzle" style="width: 100%; border-radius: 5px;">
            </div>
            <div style="text-align: center;">
                <button class="calibrate-btn btn" data-variant-idx="${idx}" style="background: #2196F3; padding: 8px 16px; font-size: 0.9em; margin-right: 8px;">
                    📐 Kalibrieren
                </button>
                <button class="regenerate-btn btn" data-seed="${metadata.seed}" style="background: #FF9800; padding: 8px 16px; font-size: 0.9em;">
                    🔄 Mit Seed neu generieren
                </button>
            </div>
        `;

        // Add click handler to select variant
        variantCard.addEventListener('click', function() {
            // Remove selection from all cards
            document.querySelectorAll('#variantsGrid .piece-card').forEach(card => {
                card.style.border = '3px solid transparent';
                card.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
            });

            // Mark this card as selected
            variantCard.style.border = '3px solid #4CAF50';
            variantCard.style.boxShadow = '0 4px 15px rgba(76, 175, 80, 0.4)';

            // Convert base64 to blob
            const base64Data = variant.noisy_image;
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            selectedVariantBlob = new Blob([byteArray], { type: 'image/png' });
            selectedVariantSeed = metadata.seed;

            // Enable global clean and extract buttons
            cleanGlobalBtn.disabled = false;
            extractBtn.disabled = false;
            cleanPiecesBtn.disabled = true;
            cornersBtn.disabled = true;
            matchingBtn.disabled = true;
            uploadedFilename = null;

            showStatus(`Variante ${idx + 1} ausgewählt (Seed: ${metadata.seed}). Klicke auf "2 - Global säubern" oder "3 - Teile extrahieren".`, 'success');
        });

        // Add hover effect
        variantCard.addEventListener('mouseenter', function() {
            if (variantCard.style.border !== '3px solid #4CAF50') {
                variantCard.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.2)';
            }
        });
        variantCard.addEventListener('mouseleave', function() {
            if (variantCard.style.border !== '3px solid #4CAF50') {
                variantCard.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
            }
        });

        // Add click handler for calibration button
        const calibrateBtn = variantCard.querySelector('.calibrate-btn');
        calibrateBtn.addEventListener('click', async function(e) {
            e.stopPropagation(); // Prevent variant selection when clicking calibrate button

            const btn = e.target;
            btn.disabled = true;
            btn.textContent = '⏳ Kalibriere...';

            showStatus('Kalibriere Barrel Distortion aus Raster...', 'info');

            try {
                // Send noisy image to calibration endpoint
                const response = await fetch('/puzzle-gen/calibrate-from-grid', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: variant.noisy_image
                    })
                });

                const result = await response.json();

                if (!result.success) {
                    throw new Error(result.error || 'Calibration failed');
                }

                // Display calibration results
                const calib = result.calibration;
                showStatus(
                    `Kalibrierung erfolgreich! ` +
                    `k1=${calib.k1.toFixed(3)}, k2=${calib.k2.toFixed(3)}, k3=${calib.k3.toFixed(3)} | ` +
                    `${calib.num_lines_detected} Linien erkannt, RMS Error=${calib.rms_error.toFixed(2)}px`,
                    'success'
                );

                btn.textContent = '✓ Kalibriert';
                btn.style.background = '#4CAF50';

            } catch (error) {
                showStatus('Calibration error: ' + error.message, 'error');
                btn.textContent = '📐 Kalibrieren';
                btn.disabled = false;
            }
        });

        // click handler for regenerate button
        const regenerateBtn = variantCard.querySelector('.regenerate-btn');
        regenerateBtn.addEventListener('click', async function(e) {
            e.stopPropagation(); // Prevent variant selection

            const seed = e.target.getAttribute('data-seed');

            // Fill seed input and trigger generation
            const seedInputElem = document.getElementById('seedInput');
            seedInputElem.value = seed;

            showStatus(`Regeneriere Puzzle mit Seed ${seed}...`, 'info');

            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });

            // Wait a bit for scroll, then trigger generation
            setTimeout(() => {
                const generateBtn = document.getElementById('generateBtn');
                generateBtn.disabled = false;
                generateBtn.click();
            }, 500);
        });

        variantsGrid.appendChild(variantCard);
    });

    // Scroll to variants
    generatedVariants.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayGlobalCleanResults(data) {
    // Show results section
    globalCleanResults.classList.add('active');

    // Display calibration info in stats grid
    const statsGrid = document.getElementById('globalCleanStatsGrid');
    const calibInfo = data.calibration_info || '';
    const lines = calibInfo.split('\n');

    let statsHTML = '';
    lines.forEach(line => {
        if (line.trim()) {
            const parts = line.split(':');
            if (parts.length === 2) {
                statsHTML += `
                    <div class="stat-card">
                        <div class="value">${parts[1].trim()}</div>
                        <div class="label">${parts[0].trim()}</div>
                    </div>
                `;
            }
        }
    });

    statsGrid.innerHTML = statsHTML;

    // Display before/after comparison image
    const grid = document.getElementById('globalCleanGrid');
    if (data.images && data.images.before_after) {
        grid.innerHTML = `
            <div class="piece-card">
                <img src="/output/${data.images.before_after}" alt="Before/After Comparison"
                     style="cursor: pointer;"
                     onclick="window.open('/output/${data.images.before_after}', '_blank')">
                <div class="piece-info" style="text-align: center;">
                    <div><strong>Links:</strong> Original</div>
                    <div><strong>Rechts:</strong> Gesäubert</div>
                    <div style="color: #666; font-size: 0.9em; margin-top: 5px;">Klicke zum Vergrößern</div>
                </div>
            </div>
        `;
    } else {
        grid.innerHTML = '<p>Keine Visualisierung verfügbar</p>';
    }

    // Scroll to results
    globalCleanResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayPieceCleanResults(data) {
    // Show results section
    pieceCleanResults.classList.add('active');

    // Display statistics
    const statsGrid = document.getElementById('pieceCleanStatsGrid');
    const stats = data.statistics || {};
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="value">${data.num_pieces}</div>
            <div class="label">Gesäuberte Teile</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.avg_area || 0)}</div>
            <div class="label">Avg Area (px²)</div>
        </div>
        <div class="stat-card">
            <div class="value">${Math.round(stats.avg_perimeter || 0)}</div>
            <div class="label">Avg Perimeter (px)</div>
        </div>
    `;

    // Display contour comparison
    const comparisonGrid = document.getElementById('contourComparisonGrid');
    if (data.images && data.images.contour_comparison) {
        comparisonGrid.innerHTML = `
            <div class="piece-card">
                <img src="/output/${data.images.contour_comparison}" alt="Contour Comparison"
                     style="cursor: pointer;"
                     onclick="window.open('/output/${data.images.contour_comparison}', '_blank')">
                <div class="piece-info" style="text-align: center;">
                    <div><strong style="color: red;">Rot:</strong> Original-Konturen</div>
                    <div><strong style="color: green;">Grün:</strong> Gesäuberte Konturen</div>
                    <div style="color: #666; font-size: 0.9em; margin-top: 5px;">Klicke zum Vergrößern</div>
                </div>
            </div>
        `;
    } else {
        comparisonGrid.innerHTML = '<p>Keine Kontur-Visualisierung verfügbar</p>';
    }

    // Display cleaned pieces
    const piecesGrid = document.getElementById('cleanedPiecesGrid');
    piecesGrid.innerHTML = '';

    if (data.images && data.images.cleaned_pieces) {
        data.images.cleaned_pieces.forEach((pieceFilename, idx) => {
            const pieceCard = document.createElement('div');
            pieceCard.className = 'piece-card';
            pieceCard.innerHTML = `
                <img src="/output/${pieceFilename}" alt="Cleaned Piece ${idx}">
                <div class="piece-info">
                    <div><strong>Piece ID:</strong> ${idx}</div>
                    <div><strong>Status:</strong> Gesäubert</div>
                </div>
            `;
            piecesGrid.appendChild(pieceCard);
        });
    } else {
        piecesGrid.innerHTML = '<p>Keine gesäuberten Teile verfügbar</p>';
    }

    // Scroll to results
    pieceCleanResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
