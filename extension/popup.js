document.addEventListener('DOMContentLoaded', function() {
  const detectButton = document.getElementById('detectButton');
  if (detectButton) {
      detectButton.addEventListener('click', function() {
          chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
              chrome.tabs.sendMessage(tabs[0].id, {action: "detectHotspots"}, function(response) {
                  if (chrome.runtime.lastError) {
                      console.error("Error sending message:", chrome.runtime.lastError);
                  } else {
                      console.log("Message sent to content script.");
                      if(response) {
                          console.log("Response from content script:", response);
                      }
                  }
              });
          });
      });
  }
});