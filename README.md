# Visual Attention Analyzer

A powerful Chrome extension that analyzes webpages to detect visual hotspots, predict eye movement patterns, and visualize user attention using computer vision and DOM analysis.



## Features

- **Visual Hotspot Detection**: Identifies areas of a webpage most likely to attract user attention
- **Face Detection**: Recognizes human faces in images and prioritizes them in attention analysis
- **Eye Movement Prediction**: Simulates likely user eye movement paths across the page
- **Attention Heatmap**: Visualizes attention distribution across the entire webpage
- **DOM-Aware Analysis**: Combines visual saliency with DOM element importance for better accuracy
- **Interactive Visualization Controls**: Toggle different visualization layers on and off

## How It Works

The Visual Attention Analyzer uses a hybrid approach combining:

1. **Computer Vision Algorithms**: Spectral Residual saliency detection to identify visually distinct regions
2. **Face Detection**: MediaPipe face detection to identify human faces as high-attention areas
3. **DOM Analysis**: Examines webpage structure and element properties to determine importance
4. **Attention Modeling**: Applies psychological principles like F-pattern reading and center bias

The extension captures a screenshot of the current webpage, extracts DOM information, and sends this data to a Python-based FastAPI backend that processes the image using computer vision techniques. The results are then visualized as overlays on the webpage.

## Installation

### Prerequisites
- Chrome browser
- Python 3.8+ (for the backend server)
- Node.js and npm (optional, for development)

### Extension Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/visual-attention-analyzer.git
   ```

2. Load the extension in Chrome:
   - Open Chrome and navigate to `chrome://extensions/`
   - Enable "Developer mode" (toggle in the upper right)
   - Click "Load unpacked" and select the `extension` folder from the cloned repository

### Backend Setup
1. Install required Python packages:
   ```
   pip install fastapi uvicorn opencv-python numpy mediapipe
   ```

2. Start the backend server:
   ```
   cd visual-attention-analyzer
   python main.py
   ```
   The server will start at http://localhost:8000

## Usage

1. Navigate to any webpage you want to analyze
2. Click the Visual Attention Analyzer extension icon in your browser toolbar
3. Press the "Analyze Page Attention" button
4. View the visualizations overlaid on the webpage:
   - Red/orange/yellow boxes show attention hotspots (ranked by importance)
   - Numbered blue dots show the predicted eye movement path
   - Purple outlines indicate detected faces
   - Toggle visualizations using the control panel

## Technical Details

### Extension Components
- **background.js**: Service worker for handling browser screenshot capture
- **content.js**: Manages DOM extraction, visualization rendering, and API communications
- **popup.html/js**: User interface for triggering analysis
- **manifest.json**: Extension configuration file

### Backend Components
- **main.py**: FastAPI server implementing the computer vision algorithms

### Analysis Pipeline
1. Screenshot capture via Chrome API
2. DOM element extraction with position and style information
3. Saliency detection using Spectral Residual algorithm
4. Face detection using MediaPipe
5. DOM importance scoring based on element type, content, and position
6. Hybrid scoring combining visual and semantic importance
7. Non-maximum suppression to merge overlapping regions
8. Eye movement path prediction based on hotspot ranking
9. Attention heatmap generation

## Customization

You can modify the visualization styles in `styles.css` to change the appearance of the overlays. The three main visualization types are:

- `.hotspot-overlay`: Controls the appearance of hotspot region boxes
- `.eye-path-point` and `.eye-path-line`: Controls the appearance of eye movement path
- `.face-region`: Controls the appearance of face detection boxes

## Development

To modify the extension:
1. Make changes to the extension files
2. Reload the extension in Chrome's extension management page

To modify the backend:
1. Update the algorithms in `main.py`
2. The server will automatically reload if you started it with the `reload=True` parameter

## Limitations

- The analyzer works best on pages that are fully loaded
- Some websites may block content scripts due to security policies
- Analysis may take a few seconds on complex pages with many elements
- Face detection requires visible human faces in the viewport

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision algorithms
- MediaPipe for face detection capabilities
- FastAPI for the efficient backend framework