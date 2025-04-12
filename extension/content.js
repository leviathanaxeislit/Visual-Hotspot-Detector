console.log("Enhanced Visual Attention Analyzer loaded");

// Function to fetch hotspots from the API
async function fetchHotspots() {
    console.log("Fetching hotspots - requesting screenshot and DOM data...");

    chrome.runtime.sendMessage({ action: "captureScreenshot" }, async function(response) {
        if (chrome.runtime.lastError || response.error) {
            console.error("Error capturing screenshot from background script:", chrome.runtime.lastError || response.error);
            return;
        }

        const dataUrl = response.dataUrl;
        const base64ImageData = dataUrl.split(',')[1];

        // --- Extract Enhanced DOM Data ---
        const domData = extractEnhancedDOMData();
        console.log(`Extracted DOM Data: ${domData.length} elements`);

        // Get viewport dimensions for better analysis
        const viewportSize = {
            width: window.innerWidth,
            height: window.innerHeight
        };

        // Send base64 image, DOM data, and viewport info to FastAPI backend
        try {
            const apiResponse = await fetch('http://localhost:8000/hotspots', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: base64ImageData,
                    dom_data: domData,
                    viewport_size: viewportSize
                })
            });

            if (!apiResponse.ok) {
                throw new Error(`HTTP error! status: ${apiResponse.status}`);
            }

            const data = await apiResponse.json();
            console.log("API Response received with features:", Object.keys(data));

            // Create control panel for visualization options
            createControlPanel();

            // Display visualization elements based on API response
            if (data.hotspot_regions && data.hotspot_regions.length > 0) {
                console.log(`Received ${data.hotspot_regions.length} hotspot regions.`);
                
                // Convert string keys back to tuple arrays for scores
                const processedScores = {};
                if (data.hotspot_scores) {
                    Object.keys(data.hotspot_scores).forEach(key => {
                        // Parse string representation of array back to actual array
                        const coords = JSON.parse(key.replace(/\(/g, '[').replace(/\)/g, ']').replace(/,\s*/g, ','));
                        processedScores[key] = data.hotspot_scores[key];
                    });
                }

                // Display hotspot regions with colored overlays
                displayHotspotRegions(data.hotspot_regions, processedScores);
                
                // Display attention heatmap if available
                if (data.attention_heatmap_base64) {
                    displayAttentionHeatmap(data.attention_heatmap_base64);
                }
                
                // Display eye movement path if available
                if (data.eye_movement_path && data.eye_movement_path.length > 0) {
                    displayEyeMovementPath(data.eye_movement_path);
                }
                
                // Display face regions if available
                if (data.face_regions && data.face_regions.length > 0) {
                    displayFaceRegions(data.face_regions);
                }
            } else {
                console.log("No hotspot regions received from API.");
                clearAllOverlays();
            }

        } catch (error) {
            console.error("Error fetching hotspots:", error);
        }
    });
}

// Function to extract enhanced DOM data with more attributes
function extractEnhancedDOMData() {
    const importantElements = [];
    const elementsToConsider = document.querySelectorAll('input, button, a, h1, h2, h3, h4, h5, h6, p, nav, ul, ol, li, form, textarea, select, img, video, div, span, label, header, footer');

    elementsToConsider.forEach(element => {
        const rect = element.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0 && isElementVisible(element)) {
            // Extract element attributes
            const attributes = {};
            for (const attr of element.attributes) {
                attributes[attr.name] = attr.value;
            }
            
            // Get computed style values
            const computedStyle = window.getComputedStyle(element);
            
            importantElements.push({
                tag_name: element.tagName.toLowerCase(),
                bounding_box: [rect.left, rect.top, rect.right, rect.bottom],
                text_content: element.textContent.trim().substring(0, 200),
                attributes: attributes,
                style: {
                    backgroundColor: computedStyle.backgroundColor,
                    color: computedStyle.color,
                    fontSize: computedStyle.fontSize,
                    fontWeight: computedStyle.fontWeight,
                    display: computedStyle.display,
                    position: computedStyle.position,
                    zIndex: computedStyle.zIndex
                },
                is_interactive: isInteractiveElement(element),
                dom_depth: getDOMDepth(element)
            });
        }
    });
    return importantElements;
}

// Helper function to check if element is visible
function isElementVisible(element) {
    const style = window.getComputedStyle(element);
    return style.display !== 'none' && 
           style.visibility !== 'hidden' && 
           style.opacity !== '0';
}

// Helper function to check if element is interactive
function isInteractiveElement(element) {
    const tag = element.tagName.toLowerCase();
    if (['a', 'button', 'input', 'select', 'textarea'].includes(tag)) {
        return true;
    }
    
    // Check for click handlers
    const clickEvents = getEventListeners(element)?.click;
    if (clickEvents && clickEvents.length > 0) {
        return true;
    }
    
    // Check for interactive attributes
    return element.getAttribute('onclick') !== null || 
           element.getAttribute('role') === 'button' ||
           element.getAttribute('tabindex') !== null;
}

// Since getEventListeners is only available in Chrome DevTools, 
// we'll use a simplified approach
function getEventListeners(element) {
    // This is a simplified implementation since we can't directly access events in content scripts
    // In a real implementation, you might use a MutationObserver to track elements with event handlers
    return null;
}

// Calculate DOM depth (distance from root)
function getDOMDepth(element) {
    let depth = 0;
    let current = element;
    while (current.parentNode) {
        depth++;
        current = current.parentNode;
    }
    return depth;
}

// Create a control panel to toggle visualization options
function createControlPanel() {
    clearControlPanel();
    
    const controlPanel = document.createElement('div');
    controlPanel.id = 'attention-analyzer-controls';
    controlPanel.style.position = 'fixed';
    controlPanel.style.top = '10px';
    controlPanel.style.right = '10px';
    controlPanel.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    controlPanel.style.padding = '10px';
    controlPanel.style.borderRadius = '5px';
    controlPanel.style.zIndex = '10000';
    controlPanel.style.color = 'white';
    controlPanel.style.fontFamily = 'Arial, sans-serif';
    controlPanel.style.fontSize = '12px';
    controlPanel.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.5)';
    
    // Add title
    const title = document.createElement('div');
    title.textContent = 'Visual Attention Analyzer';
    title.style.fontWeight = 'bold';
    title.style.marginBottom = '8px';
    title.style.textAlign = 'center';
    controlPanel.appendChild(title);
    
    // Add toggle options
    const options = [
        { id: 'toggle-hotspots', label: 'Hotspot Regions', checked: true },
        { id: 'toggle-heatmap', label: 'Attention Heatmap', checked: false },
        { id: 'toggle-eyepath', label: 'Eye Movement Path', checked: true },
        { id: 'toggle-faces', label: 'Face Detection', checked: true }
    ];
    
    options.forEach(option => {
        const label = document.createElement('label');
        label.style.display = 'block';
        label.style.margin = '5px 0';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = option.id;
        checkbox.checked = option.checked;
        checkbox.addEventListener('change', toggleVisualization);
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(' ' + option.label));
        controlPanel.appendChild(label);
    });
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.textContent = 'Close';
    closeButton.style.display = 'block';
    closeButton.style.width = '100%';
    closeButton.style.marginTop = '10px';
    closeButton.style.padding = '3px';
    closeButton.style.backgroundColor = '#333';
    closeButton.style.color = 'white';
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '3px';
    closeButton.style.cursor = 'pointer';
    closeButton.addEventListener('click', () => {
        clearAllOverlays();
        controlPanel.remove();
    });
    controlPanel.appendChild(closeButton);
    
    document.body.appendChild(controlPanel);
}

// Toggle visualization elements based on checkbox state
function toggleVisualization(event) {
    const id = event.target.id;
    const checked = event.target.checked;
    
    switch(id) {
        case 'toggle-hotspots':
            document.querySelectorAll('.hotspot-overlay').forEach(el => {
                el.style.display = checked ? 'block' : 'none';
            });
            break;
        case 'toggle-heatmap':
            const heatmap = document.getElementById('attention-heatmap');
            if (heatmap) heatmap.style.display = checked ? 'block' : 'none';
            break;
        case 'toggle-eyepath':
            document.querySelectorAll('.eye-path-point, .eye-path-line').forEach(el => {
                el.style.display = checked ? 'block' : 'none';
            });
            break;
        case 'toggle-faces':
            document.querySelectorAll('.face-region').forEach(el => {
                el.style.display = checked ? 'block' : 'none';
            });
            break;
    }
}

// Display hotspot regions with color coding based on importance
function displayHotspotRegions(regions, scores) {
    clearOverlays('hotspot-overlay');
    
    regions.forEach((region, index) => {
        const [x1, y1, x2, y2] = region;
        const hotspotDiv = document.createElement('div');
        hotspotDiv.className = 'hotspot-overlay';
        hotspotDiv.style.position = 'absolute';
        hotspotDiv.style.left = `${x1}px`;
        hotspotDiv.style.top = `${y1}px`;
        hotspotDiv.style.width = `${x2 - x1}px`;
        hotspotDiv.style.height = `${y2 - y1}px`;
        hotspotDiv.style.pointerEvents = 'none';
        hotspotDiv.style.zIndex = '9998';
        hotspotDiv.style.transition = 'all 0.3s ease-in-out';
        
        // Color coding based on rank (importance)
        let color, borderColor, opacity;
        
        if (index < 3) { // Top 3 hotspots - Red with higher opacity
            color = 'red';
            borderColor = 'darkred';
            opacity = 0.5;
        } else if (index < 8) { // Next 5 hotspots - Orange
            color = 'orange';
            borderColor = 'darkorange';
            opacity = 0.4;
        } else { // Rest - Yellow with lower opacity
            color = 'yellow';
            borderColor = 'gold';
            opacity = 0.3;
        }
        
        hotspotDiv.style.backgroundColor = `${color}`;
        hotspotDiv.style.opacity = `${opacity}`;
        hotspotDiv.style.border = `2px solid ${borderColor}`;
        
        // Add rank number for top 10 hotspots
        if (index < 10) {
            const rankLabel = document.createElement('div');
            rankLabel.textContent = (index + 1).toString();
            rankLabel.style.position = 'absolute';
            rankLabel.style.top = '2px';
            rankLabel.style.left = '2px';
            rankLabel.style.backgroundColor = borderColor;
            rankLabel.style.color = 'white';
            rankLabel.style.padding = '2px 6px';
            rankLabel.style.borderRadius = '50%';
            rankLabel.style.fontSize = '12px';
            rankLabel.style.fontWeight = 'bold';
            hotspotDiv.appendChild(rankLabel);
        }
        
        document.body.appendChild(hotspotDiv);
    });
}

// Display attention heatmap as semi-transparent overlay
function displayAttentionHeatmap(heatmapBase64) {
    clearOverlays('attention-heatmap');
    
    const heatmapImg = document.createElement('img');
    heatmapImg.id = 'attention-heatmap';
    heatmapImg.src = `data:image/png;base64,${heatmapBase64}`;
    heatmapImg.style.position = 'absolute';
    heatmapImg.style.top = '0';
    heatmapImg.style.left = '0';
    heatmapImg.style.width = '100%';
    heatmapImg.style.height = '100%';
    heatmapImg.style.objectFit = 'cover';
    heatmapImg.style.opacity = '0.7';
    heatmapImg.style.pointerEvents = 'none';
    heatmapImg.style.zIndex = '9997';
    heatmapImg.style.display = 'none'; // Hidden by default, toggle with control panel
    
    document.body.appendChild(heatmapImg);
}

// Display predicted eye movement path with numbered points and connecting lines
function displayEyeMovementPath(pathPoints) {
    clearOverlays('eye-path-point');
    clearOverlays('eye-path-line');
    
    // Create SVG container for path visualization
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.zIndex = '9999';
    svg.style.pointerEvents = 'none';
    svg.classList.add('eye-path-line');
    
    // Draw lines connecting path points
    for (let i = 0; i < pathPoints.length - 1; i++) {
        const [x1, y1] = pathPoints[i];
        const [x2, y2] = pathPoints[i + 1];
        
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', '#00BFFF'); // Bright blue
        line.setAttribute('stroke-width', '3');
        line.setAttribute('stroke-dasharray', '5,5'); // Dashed line
        line.setAttribute('marker-end', 'url(#arrow)');
        
        svg.appendChild(line);
    }
    
    // Add arrow marker definition
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrow');
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '5');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '6');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('orient', 'auto-start-reverse');
    
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
    path.setAttribute('fill', '#00BFFF');
    
    marker.appendChild(path);
    defs.appendChild(marker);
    svg.appendChild(defs);
    
    document.body.appendChild(svg);
    
    // Create numbered points for eye path
    pathPoints.forEach((point, index) => {
        const [x, y] = point;
        
        const pointDiv = document.createElement('div');
        pointDiv.className = 'eye-path-point';
        pointDiv.style.position = 'absolute';
        pointDiv.style.left = `${x - 12}px`;
        pointDiv.style.top = `${y - 12}px`;
        pointDiv.style.width = '24px';
        pointDiv.style.height = '24px';
        pointDiv.style.borderRadius = '50%';
        pointDiv.style.backgroundColor = index === 0 ? '#32CD32' : '#00BFFF'; // Start point is green
        pointDiv.style.color = 'white';
        pointDiv.style.textAlign = 'center';
        pointDiv.style.lineHeight = '24px';
        pointDiv.style.fontSize = '12px';
        pointDiv.style.fontWeight = 'bold';
        pointDiv.style.zIndex = '10000';
        pointDiv.style.pointerEvents = 'none';
        pointDiv.style.boxShadow = '0 0 5px rgba(0,0,0,0.5)';
        pointDiv.textContent = (index + 1).toString();
        
        // Animated pulse effect for path points
        pointDiv.style.animation = 'pulse 2s infinite';
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.2); opacity: 0.7; }
                100% { transform: scale(1); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(pointDiv);
    });
}

// Display face detection regions
function displayFaceRegions(faces) {
    clearOverlays('face-region');
    
    faces.forEach(face => {
        const [x1, y1, x2, y2] = face;
        
        const faceDiv = document.createElement('div');
        faceDiv.className = 'face-region';
        faceDiv.style.position = 'absolute';
        faceDiv.style.left = `${x1}px`;
        faceDiv.style.top = `${y1}px`;
        faceDiv.style.width = `${x2 - x1}px`;
        faceDiv.style.height = `${y2 - y1}px`;
        faceDiv.style.border = '3px solid #FF00FF'; // Magenta
        faceDiv.style.borderRadius = '5px';
        faceDiv.style.pointerEvents = 'none';
        faceDiv.style.zIndex = '9999';
        
        // Add face icon
        const faceIcon = document.createElement('div');
        faceIcon.style.position = 'absolute';
        faceIcon.style.top = '-20px';
        faceIcon.style.right = '0';
        faceIcon.style.backgroundColor = '#FF00FF';
        faceIcon.style.color = 'white';
        faceIcon.style.padding = '2px 6px';
        faceIcon.style.borderRadius = '3px';
        faceIcon.style.fontSize = '10px';
        faceIcon.innerHTML = '👤 Face';
        
        faceDiv.appendChild(faceIcon);
        document.body.appendChild(faceDiv);
    });
}

// Clear overlays by class name
function clearOverlays(className) {
    const existingOverlays = document.querySelectorAll('.' + className);
    existingOverlays.forEach(overlay => overlay.remove());
}

// Clear control panel
function clearControlPanel() {
    const existingPanel = document.getElementById('attention-analyzer-controls');
    if (existingPanel) existingPanel.remove();
}

// Clear all visual elements
function clearAllOverlays() {
    clearOverlays('hotspot-overlay');
    clearOverlays('eye-path-point');
    clearOverlays('eye-path-line');
    clearOverlays('face-region');
    
    const heatmap = document.getElementById('attention-heatmap');
    if (heatmap) heatmap.remove();
}

// Listen for messages from popup or background script
chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.action === "detectHotspots") {
            console.log("Detect hotspots action received in content script.");
            fetchHotspots();
            sendResponse({success: true});
            return true;
        } else if (request.action === "clearVisualizations") {
            clearAllOverlays();
            clearControlPanel();
            sendResponse({success: true});
            return true;
        }
    }
);

console.log("Enhanced Visual Attention Analyzer is ready to analyze the page");