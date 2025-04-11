console.log("Hotspot analysis initiated.");

function computeAdvancedAttentionScore(element) {
  const rect = element.getBoundingClientRect();
  if (!element.offsetParent || rect.width < 10 || rect.height < 10) return 0;

  const styles = window.getComputedStyle(element);
  const tag = element.tagName.toUpperCase();

  const size = rect.width * rect.height;
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const centerBias = 1 - (
    Math.abs(centerX - window.innerWidth / 2) / (window.innerWidth / 2) +
    Math.abs(centerY - window.innerHeight / 2) / (window.innerHeight / 2)
  ) / 2;

  function hexToRgb(hex) {
    const match = hex.replace('#', '').match(/.{1,2}/g);
    return match ? match.map(x => parseInt(x, 16)) : [255, 255, 255];
  }

  function getLuminance(r, g, b) {
    const a = [r, g, b].map(v => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
  }

  function getContrastScore(fg, bg) {
    const rgbFg = hexToRgb(fg);
    const rgbBg = hexToRgb(bg);
    const lum1 = getLuminance(...rgbFg);
    const lum2 = getLuminance(...rgbBg);
    const contrastRatio = (Math.max(lum1, lum2) + 0.05) / (Math.min(lum1, lum2) + 0.05);
    return Math.min(1, contrastRatio / 21);
  }

  const contrast = getContrastScore(styles.color, styles.backgroundColor);
  const fontSize = parseFloat(styles.fontSize) || 0;
  const fontWeight = styles.fontWeight === 'bold' || parseInt(styles.fontWeight) > 600 ? 1 : 0;
  const visibilityScore = (centerBias + contrast + fontSize / 100 + fontWeight) / 4;

  const text = element.innerText || '';
  const wordCount = text.split(/\s+/).length;
  const isClickable = ['A', 'BUTTON', 'INPUT'].includes(tag) || styles.cursor === 'pointer';

  const tagPriority = {
    H1: 1, H2: 0.95, H3: 0.9, IMG: 0.85, A: 0.8,
    BUTTON: 0.8, LABEL: 0.7, SPAN: 0.4, P: 0.6, DIV: 0.3
  }[tag] || 0.2;

  const densityPenalty = wordCount > 50 ? 0.5 : 1;
  const semanticScore = (tagPriority + (isClickable ? 0.5 : 0) + densityPenalty) / 3;

  const yBias = rect.top / window.innerHeight;
  const fixationScore = 1 - yBias;

  const finalScore = (
    0.4 * visibilityScore +
    0.35 * semanticScore +
    0.25 * fixationScore
  );

  return finalScore;
}

function highlightHotspots() {
    const all = [...document.body.querySelectorAll('*')];
    console.log(`Total elements scanned: ${all.length}`);
  
    let count = 0;
  
    all.forEach((el) => {
      const score = computeAdvancedAttentionScore(el);
      const rect = el.getBoundingClientRect();
  
      // Log scores
      console.log(`[${el.tagName}] Score: ${score.toFixed(2)} | Text: "${el.innerText?.slice(0, 30)}"`);
  
      // Lower threshold for testing
      if (score > 0.3 && rect.width > 5 && rect.height > 5) {
        count++;
  
        const overlay = document.createElement("div");
        overlay.style.position = "absolute";
        overlay.style.left = `${rect.left + window.scrollX}px`;
        overlay.style.top = `${rect.top + window.scrollY}px`;
        overlay.style.width = `${rect.width}px`;
        overlay.style.height = `${rect.height}px`;
        overlay.style.backgroundColor = `rgba(255, 0, 0, ${Math.min(score, 0.7)})`;
        overlay.style.border = "1px solid yellow";
        overlay.style.pointerEvents = "none";
        overlay.style.zIndex = 2147483647;
  
        document.body.appendChild(overlay);
      }
    });
  
    console.log(`🔥 Hotspots highlighted: ${count}`);
  
    // Add green test box to verify overlays render
    const testBox = document.createElement("div");
    testBox.style.position = "absolute";
    testBox.style.left = "100px";
    testBox.style.top = "100px";
    testBox.style.width = "100px";
    testBox.style.height = "100px";
    testBox.style.backgroundColor = "rgba(0, 255, 0, 0.4)";
    testBox.style.zIndex = 2147483647;
    testBox.style.pointerEvents = "none";
    testBox.textContent = "Test Box";
    document.body.appendChild(testBox);
  }
  
highlightHotspots();
