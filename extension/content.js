console.log("Content script loaded");

// Function to fetch hotspots from the API
async function fetchHotspots() {
    console.log("Fetching hotspots - requesting screenshot from background script...");

    chrome.runtime.sendMessage({ action: "captureScreenshot" }, async function(response) {
        if (chrome.runtime.lastError || response.error) {
            console.error("Error capturing screenshot from background script:", chrome.runtime.lastError || response.error);
            return;
        }

        const dataUrl = response.dataUrl;
        const base64ImageData = dataUrl.split(',')[1];

        // Send base64 image to FastAPI backend
        try {
            const apiResponse = await fetch('http://localhost:8000/hotspots', { // Adjust URL if needed
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: base64ImageData })
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
                // Optionally, clear any existing overlays if no hotspots are found
                clearHotspotOverlays();
            }

        } catch (error) {
            console.error("Error fetching hotspots:", error);
        }
    });
}

// Function to display hotspot regions on the page
function displayHotspotRegions(regions) {
    // Clear any existing hotspot overlays first
    clearHotspotOverlays();

    // Create and append a div for each hotspot region
    regions.forEach(region => {
        const [x1, y1, x2, y2] = region;
        const hotspotDiv = document.createElement('div');

        hotspotDiv.className = 'hotspot-overlay'; // Add a class for styling
        hotspotDiv.style.position = 'absolute';
        hotspotDiv.style.left = `${x1}px`;
        hotspotDiv.style.top = `${y1}px`;
        hotspotDiv.style.width = `${x2 - x1}px`;
        hotspotDiv.style.height = `${y2 - y1}px`;
        hotspotDiv.style.backgroundColor = 'rgba(255, 0, 0, 0.4)'; // Example: Semi-transparent red
        hotspotDiv.style.border = '2px solid red'; // Example: Red border
        hotspotDiv.style.zIndex = '1001'; // Ensure it's above saliency map if both displayed
        hotspotDiv.style.pointerEvents = 'none'; // Make non-interactive

        document.body.appendChild(hotspotDiv);
    });
    console.log("Hotspot region overlays appended.");
}

// Function to clear existing hotspot overlays
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
            sendResponse({}); // Send empty response back to popup (optional but good practice)
            return true; // Indicate you wish to send a response asynchronously
        }
    }
);

console.log("Content script is ready to receive messages.");