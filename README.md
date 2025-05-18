# Visual Attention Analyzer

A powerful Chrome extension that analyzes webpages to detect visual hotspots, predict eye movement patterns, and visualize user attention using computer vision and DOM analysis.

## Demo Link

### Youtube Video Link:
[<img src="https://i.ytimg.com/vi/L-xTkwNaORw/maxresdefault.jpg">](https://www.youtube.com/watch?v=L-xTkwNaORw)

## Presentation

[Download or View Presentation ](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2Fleviathanaxeislit%2FVisual-Hotspot-Detector%2F93cf870c7c824216280dab35b8744537bcca3e53%2Fdocs%2FClarityUX%2520Assessment%2520PPT.pptx&wdOrigin=BROWSELINK)

## Features

- **Visual Hotspot Detection**: Identifies areas of a webpage most likely to attract user attention
- **Face Detection**: Recognizes human faces in images and prioritizes them in attention analysis
- **Eye Movement Prediction**: Simulates likely user eye movement paths across the page
- **Attention Heatmap**: Visualizes attention distribution across the entire webpage
- **DOM-Aware Analysis**: Combines visual saliency with DOM element importance for better accuracy
- **Figma Design Analysis**: Analyzes exported Figma frames with layer data to predict visual attention on designs before development.
- **Interactive Visualization Controls**: Toggle different visualization layers on and off

## How It Works

The Visual Attention Analyzer uses a hybrid approach combining:

1. **Computer Vision Algorithms**: Spectral Residual saliency detection to identify visually distinct regions
2. **Face Detection**: MediaPipe face detection to identify human faces as high-attention areas
3. **DOM Analysis (for webpages)**: Examines webpage structure and element properties to determine importance.
4. **Figma Layer Analysis (for Figma designs)**: Parses exported Figma layer data (type, bounding box, properties) to assess element importance within a design.
5. **Attention Modeling**: Applies psychological principles like F-pattern reading and center bias

The extension captures a screenshot of the current webpage, extracts DOM information, and sends this data to a Python-based FastAPI backend that processes the image using computer vision techniques. For Figma designs, an exported image and corresponding layer data are sent to a dedicated backend endpoint for analysis. The results are then visualized as overlays on the webpage (for live analysis) or returned as structured data (for Figma analysis).

## Installation

### Prerequisites
- Chrome browser
- Python 3.12 (for the backend server)
- 

### Extension Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/leviathanaxeislit/Visual-Hotspot-Detector.git
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
   ```bash
   cd Visual-Hotspot-Detector
   python main.py
   ```
   The server will start at http://localhost:8000

## Usage

### Analyzing Live Webpages
1. Navigate to any webpage you want to analyze
2. Click the Visual Attention Analyzer extension icon in your browser toolbar
3. Press the "Analyze Page Attention" button
4. View the visualizations overlaid on the webpage:
   - Red/orange/yellow boxes show attention hotspots (ranked by importance)
   - Numbered blue dots show the predicted eye movement path
   - Purple outlines indicate detected faces
   - Toggle visualizations using the control panel

### Analyzing Figma Designs
The system also supports analyzing exported Figma frames. This requires an image of the frame and its corresponding layer data.

1.  **Export your Figma frame as an image** (e.g., PNG).
2.  **Extract layer data from your Figma design.** This data should be in a JSON format, where each element includes properties like `id`, `name`, `type`, `bounding_box`, `text_content`, `visible`, `opacity`, etc. (Refer to the `FigmaElement` model in `main.py` for the expected structure).
3.  **Send the image (as base64) and layer data to the backend API.** You can use a tool like `curl` or a custom script.

    Example `curl` command:
    ```bash
    curl -X POST "http://localhost:8000/figma_visual_analysis" \
         -H "Content-Type: application/json" \
         -d '{
               "image_base64": "YOUR_BASE64_ENCODED_IMAGE_STRING",
               "figma_layer_data": [
                 { "id": "1:2", "name": "Header Frame", "type": "FRAME", "bounding_box": [0, 0, 1280, 100], "...other_props" },
                 { "id": "1:3", "name": "CTA Button", "type": "RECTANGLE", "text_content": "Click Me", "bounding_box": [500, 400, 700, 450], "...other_props" }
                 // ... more layer objects
               ],
               "viewport_size": {"width": 1280, "height": 800} // Dimensions of the exported frame image
             }'
    ```
4.  **Interpret the JSON response.** The API will return a JSON object containing:
    *   `saliency_map_base64`: Base64 encoded saliency map image.
    *   `attention_heatmap_base64`: Base64 encoded attention heatmap image.
    *   `hotspot_regions`: List of detected hotspot bounding boxes.
    *   `face_regions`: List of detected face bounding boxes.
    *   `eye_movement_path`: Predicted eye movement path.
    *   `dom_importance` (renamed to `figma_layer_importance` in this context): Scores for individual Figma layers.
    *   `hotspot_scores`: Scores for each hotspot region.

    This data can be used to programmatically understand attention patterns or to generate visualizations.

## Demo

<image src="/images/screenshot-1744540472449.png"></image>
<image src="/images/screenshot-1744540562980.png"></image>
<image src="/images/Screenshot (633).png"></image>


## System Architecture

<image src="/images/Architecture.png"></image>

## Technical Details

### Extension Components
- **background.js**: Service worker for handling browser screenshot capture
- **content.js**: Manages DOM extraction, visualization rendering, and API communications
- **popup.html/js**: User interface for triggering analysis
- **manifest.json**: Extension configuration file

### Backend Components
- **main.py**: FastAPI server implementing the computer vision algorithms.
    -   `/hotspots`: Endpoint for live webpage analysis.
    -   `/figma_visual_analysis`: Endpoint for Figma design analysis.

### Analysis Pipeline
1. Screenshot capture via Chrome API (for live webpages) or image provision (for Figma).
2. DOM element extraction (for live webpages) or Figma layer data parsing (for Figma designs).
3. Saliency detection using Spectral Residual algorithm.
4. Face detection using MediaPipe.
5. DOM importance scoring (for live webpages) or Figma layer importance scoring (for Figma designs) based on element type, content, and properties.
6. Hybrid scoring combining visual and semantic importance.
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

## Acknowledgments

- OpenCV for computer vision algorithms
- MediaPipe for face detection capabilities
- FastAPI for the efficient backend framework