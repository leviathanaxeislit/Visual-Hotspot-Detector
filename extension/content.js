console.log("Content script loaded");

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

        // --- Extract DOM Data ---
        const domData = extractDOMData(); // Call function to get DOM data
        console.log("Extracted DOM Data:", domData); // Log the extracted DOM data

        // Send base64 image AND DOM data to FastAPI backend
        try {
            const apiResponse = await fetch('http://localhost:8000/hotspots', { // Adjust URL if needed
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: base64ImageData,
                    dom_data: domData // Send DOM data in the request body
                })
            });

            if (!apiResponse.ok) {
                throw new Error(`HTTP error! status: ${apiResponse.status}`);
            }

            const data = await apiResponse.json();
            console.log("API Response:", data);

            // --- Display Hotspot Regions ---
            if (data.hotspot_regions && data.hotspot_regions.length > 0) {
                console.log(`Received ${data.hotspot_regions.length} hotspot regions.`);
                displayHotspotRegions(data.hotspot_regions);
            } else {
                console.log("No hotspot regions received from API.");
                clearHotspotOverlays();
            }

        } catch (error) {
            console.error("Error fetching hotspots:", error);
        }
    });
}

// Function to extract relevant DOM data
function extractDOMData() {
    const importantElements = [];
    const elementsToConsider = document.querySelectorAll('input, button, a, h1, h2, h3, h4, h5, h6, p, nav, ul, ol, li, form, textarea, select'); // Add more selectors as needed

    elementsToConsider.forEach(element => {
        const rect = element.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) { // Ensure element is visible (has dimensions)
            importantElements.push({
                tag_name: element.tagName.toLowerCase(),
                bounding_box: [rect.left, rect.top, rect.right, rect.bottom],
                text_content: element.textContent.trim().substring(0, 200) // Get first 200 chars of text content
                // You can extract more attributes if needed (e.g., classes, ids, href for links, etc.)
            });
        }
    });
    return importantElements;
}


function displayHotspotRegions(regions) {
    clearHotspotOverlays();
    regions.forEach(region => {
        const [x1, y1, x2, y2] = region;
        const hotspotDiv = document.createElement('div');
        hotspotDiv.className = 'hotspot-overlay';
        hotspotDiv.style.position = 'absolute';
        hotspotDiv.style.left = `${x1}px`;
        hotspotDiv.style.top = `${y1}px`;
        hotspotDiv.style.width = `${x2 - x1}px`;
        hotspotDiv.style.height = `${y2 - y1}px`;
        hotspotDiv.style.backgroundColor = 'rgba(255, 0, 0, 0.4)';
        hotspotDiv.style.border = '1px solid red';
        hotspotDiv.style.zIndex = '1001';
        hotspotDiv.style.pointerEvents = 'none';
        document.body.appendChild(hotspotDiv);
    });
    console.log("Hotspot region overlays appended.");
}

function clearHotspotOverlays() {
    const existingOverlays = document.querySelectorAll('.hotspot-overlay');
    existingOverlays.forEach(overlay => overlay.remove());
    console.log("Existing hotspot overlays cleared.");
}


// Listen for messages from popup or background script to trigger hotspot detection
chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.action === "detectHotspots") {
            console.log("Detect hotspots action received in content script.");
            fetchHotspots();
            sendResponse({});
            return true;
        }
    }
);

console.log("Content script is ready to receive messages.");