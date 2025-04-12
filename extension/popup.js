document.addEventListener('DOMContentLoaded', function () {
  const detectButton = document.getElementById('detectButton');
  const clearButton = document.getElementById('clearButton');
  const statusEl = document.getElementById('status');
  const spinnerEl = document.getElementById('spinner');

  // Function to update status message with optional styling
  function updateStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.style.color = isError ? '#e74c3c' : '#555';
    
    if (isError) {
      statusEl.style.backgroundColor = '#fdedec';
      statusEl.style.padding = '8px';
      statusEl.style.borderRadius = '4px';
    } else {
      statusEl.style.backgroundColor = 'transparent';
      statusEl.style.padding = '0';
    }
  }

  // Function to show/hide loading spinner
  function toggleSpinner(show) {
    spinnerEl.style.display = show ? 'block' : 'none';
  }

  if (detectButton) {
    detectButton.addEventListener('click', function () {
      // Update UI to show loading state
      updateStatus("Analyzing page attention patterns...");
      toggleSpinner(true);
      
      // Disable button during analysis
      detectButton.disabled = true;
      detectButton.style.opacity = '0.7';
      detectButton.style.cursor = 'not-allowed';

      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (!tabs || tabs.length === 0) {
          updateStatus("No active tab found.", true);
          toggleSpinner(false);
          detectButton.disabled = false;
          detectButton.style.opacity = '1';
          detectButton.style.cursor = 'pointer';
          return;
        }

        chrome.tabs.sendMessage(tabs[0].id, { action: "detectHotspots" }, function (response) {
          // Re-enable button
          detectButton.disabled = false;
          detectButton.style.opacity = '1';
          detectButton.style.cursor = 'pointer';
          
          if (chrome.runtime.lastError) {
            console.error("Error sending message:", chrome.runtime.lastError.message);
            updateStatus("Error: Could not analyze this page. Please refresh and try again.", true);
            toggleSpinner(false);
          } else {
            console.log("Message sent to content script.");
            if (response && response.success) {
              updateStatus("Analysis complete! Visualizations displayed ✅");
            } else {
              updateStatus("Analysis complete, but with some issues ⚠️");
            }
            toggleSpinner(false);
          }
        });
      });
    });
  }
  
  if (clearButton) {
    clearButton.addEventListener('click', function() {
      chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        if (tabs && tabs.length > 0) {
          chrome.tabs.sendMessage(tabs[0].id, { action: "clearVisualizations" }, function(response) {
            if (response && response.success) {
              updateStatus("Visualizations cleared");
            }
          });
        }
      });
    });
  }
});