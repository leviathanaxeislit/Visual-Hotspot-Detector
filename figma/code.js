figma.showUI(__html__, { width: 300, height: 200 });

const FIGMA_LAYER_TYPE_MAP = {
  FRAME: 'frame',
  GROUP: 'group',
  RECTANGLE: 'rectangle',
  TEXT: 'text',
  INSTANCE: 'instance',
  COMPONENT: 'component',
  VECTOR: 'vector',
  ELLIPSE: 'ellipse',
  LINE: 'line',
  BOOLEAN_OPERATION: 'boolean_operation',
  // Add other mappings if necessary, or adjust backend to accept Figma types
};

const DEFAULT_FIGMA_TYPE = 'unknown';

figma.ui.onmessage = async (msg) => {
  if (msg.type === 'analyze-frame') {
    const selection = figma.currentPage.selection;
    if (selection.length !== 1) {
      figma.ui.postMessage({ type: 'status-update', text: 'Please select a single frame to analyze.' });
      return;
    }

    const selectedFrame = selection[0];
    if (selectedFrame.type !== 'FRAME' && selectedFrame.type !== 'COMPONENT' && selectedFrame.type !== 'INSTANCE' && selectedFrame.type !== 'GROUP') {
      figma.ui.postMessage({ type: 'status-update', text: 'Please select a frame, component, or group.' });
      return;
    }

    try {
      figma.ui.postMessage({ type: 'status-update', text: 'Exporting frame as image...' });
      const imageBytes = await selectedFrame.exportAsync({
        format: 'PNG',
        constraint: { type: 'SCALE', value: 1 }, // Use scale 1 for accurate coordinates
      });
      const imageBase64 = figma.base64Encode(imageBytes);

      figma.ui.postMessage({ type: 'status-update', text: 'Extracting layer data...' });
      const figmaLayerData = [];
      const frameX = selectedFrame.absoluteBoundingBox ? selectedFrame.absoluteBoundingBox.x : 0;
      const frameY = selectedFrame.absoluteBoundingBox ? selectedFrame.absoluteBoundingBox.y : 0;

      function extractChildren(node, parentFrameX, parentFrameY) {
        if (!node.visible) return;

        let children;
        if ('children' in node) {
          children = node.children;
        }

        if (children) {
          for (const child of children) {
            if (!child.absoluteBoundingBox) continue;

            const relX1 = child.absoluteBoundingBox.x - parentFrameX;
            const relY1 = child.absoluteBoundingBox.y - parentFrameY;
            const relX2 = relX1 + child.absoluteBoundingBox.width;
            const relY2 = relY1 + child.absoluteBoundingBox.height;

            const elementData = {
              id: child.id,
              name: child.name,
              type: FIGMA_LAYER_TYPE_MAP[child.type] || child.type.toLowerCase() || DEFAULT_FIGMA_TYPE,
              bounding_box: [relX1, relY1, relX2, relY2],
              visible: child.visible,
              opacity: 'opacity' in child ? child.opacity : 1,
              children_count: ('children' in child && child.children) ? child.children.length : 0,
            };

            if (child.type === 'TEXT' && 'characters' in child) {
              elementData.text_content = child.characters;
            }
            figmaLayerData.push(elementData);
            extractChildren(child, parentFrameX, parentFrameY);
          }
        }
      }

      extractChildren(selectedFrame, frameX, frameY);

      const requestPayload = {
        image_base64: imageBase64,
        figma_layer_data: figmaLayerData,
        viewport_size: {
          width: selectedFrame.width,
          height: selectedFrame.height,
        },
      };

      figma.ui.postMessage({ type: 'status-update', text: 'Sending data to backend...' });

      try {
        const response = await fetch('http://localhost:8000/figma_visual_analysis', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestPayload),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Backend error: ${response.status} ${errorText}`);
        }

        const results = await response.json();
        // All drawing logic on Figma canvas will be removed.
        // Instead, we send the raw data and the original frame image to the UI.

        figma.ui.postMessage({
          type: 'display-analysis',
          frameData: {
            imageBase64: imageBase64, // The base64 of the originally exported frame
            width: selectedFrame.width,
            height: selectedFrame.height
          },
          analysisResults: results // The full results object from your backend
        });

        figma.notify('Analysis sent to plugin UI for display.');

      } catch (networkError) {
        console.error('Network or backend processing error:', networkError);
        figma.ui.postMessage({ type: 'analysis-error', error: networkError.message });
        figma.notify(`Error during analysis: ${networkError.message}`, { error: true });
      }

    } catch (error) {
      console.error('Error in plugin:', error);
      figma.ui.postMessage({ type: 'analysis-error', error: error.message });
      figma.notify(`Plugin error: ${error.message}`, { error: true });
    }
  }
};