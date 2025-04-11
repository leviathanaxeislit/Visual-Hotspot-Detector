chrome.runtime.onInstalled.addListener(() => {
    console.log("Visual Hotspot Detector installed.");
  });
  
  chrome.action.onClicked.addListener((tab) => {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"]
    });
  });