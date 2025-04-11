document.addEventListener('DOMContentLoaded', function () {
  const detectButton = document.getElementById('detectButton');
  const statusEl = document.getElementById('status');
  const spinnerEl = document.getElementById('spinner');

  if (detectButton) {
    detectButton.addEventListener('click', function () {
      // Update UI to show loading state
      statusEl.textContent = "Detecting hotspots...";
      spinnerEl.style.display = "block";

      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (!tabs || tabs.length === 0) {
          statusEl.textContent = "No active tab found.";
          spinnerEl.style.display = "none";
          return;
        }

        chrome.tabs.sendMessage(tabs[0].id, { action: "detectHotspots" }, function (response) {
          if (chrome.runtime.lastError) {
            console.error("Error sending message:", chrome.runtime.lastError.message);
            statusEl.textContent = "Error: Could not contact content script.";
            spinnerEl.style.display = "none";
          } else {
            console.log("Message sent to content script.");
            if (response) {
              console.log("Response from content script:", response);
            }
            statusEl.textContent = "Hotspot detection completed ✅";
            spinnerEl.style.display = "none";
          }
        });
      });
    });
  }
});
