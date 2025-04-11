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

        // Send base64 image to FastAPI backend (rest of your fetchHotspots code remains the same)
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

            if (data.saliency_map_base64) {
                console.log("Saliency map received, displaying...");
                displaySaliencyMap(data.saliency_map_base64);
            } else {
                console.log("No saliency map data received from API.");
            }

        } catch (error) {
            console.error("Error fetching hotspots:", error);
        }
    });
}

function displaySaliencyMap(base64SaliencyMap) {
    console.log("Displaying saliency map...");
    // ... (displaySaliencyMap function - keep it the same as before) ...
    const saliencyMapImg = document.createElement('img');
    saliencyMapImg.src = `data:image/png;base64,${base64SaliencyMap}`;
    saliencyMapImg.style.position = 'absolute';
    saliencyMapImg.style.top = '0';
    saliencyMapImg.style.left = '0';
    saliencyMapImg.style.width = '100%';
    saliencyMapImg.style.height = '100%';
    saliencyMapImg.style.pointerEvents = 'none';
    saliencyMapImg.style.opacity = '0.5';
    saliencyMapImg.style.zIndex = '1000';
    document.body.appendChild(saliencyMapImg);
    console.log("Saliency map image appended to body.");
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