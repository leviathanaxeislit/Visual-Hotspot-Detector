// background.js
console.log("Background service worker loaded.");

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.action === "captureScreenshot") {
            console.log("Background script received 'captureScreenshot' action.");
            chrome.tabs.captureVisibleTab(null, { format: "png" }, function(dataUrl) {
                if (chrome.runtime.lastError) {
                    console.error("Background script captureVisibleTab error:", chrome.runtime.lastError);
                    sendResponse({ error: chrome.runtime.lastError.message }); // Send error message in response
                    return;
                }
                console.log("Background script captureVisibleTab SUCCESS, sending dataUrl back.");
                sendResponse({ dataUrl: dataUrl }); // Send dataUrl back to content script
            });
            return true; // Indicate asynchronous response
        }
        return false; // Indicate synchronous response (or no response for other actions)
    }
);

console.log("Background service worker is ready to receive messages.");