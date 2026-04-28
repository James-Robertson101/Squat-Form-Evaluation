

// State 
let currentReps    = [];
let currentRepIdx  = 0;
let jobId          = null;
let frameCount     = 0;
let currentFrame   = 0;
let playbackInterval = null;
let playbackaFps    = 15;   // controlled by speed selector

// Frame preload cache — keeps recent decoded <img> objects in memory
const CACHE_AHEAD  = 40;
const CACHE_BEHIND = 10;
const frameCache   = new Map();

//  Drag-and-drop on the upload card 
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('video-input');

dropZone.addEventListener('click', e => {
  if (e.target.classList.contains('file-label')) return; // let the label handle it
  fileInput.click();
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    setFileDisplay(file.name);
    // Inject into file input (for the FormData below)
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFileDisplay(fileInput.files[0].name);
});

function setFileDisplay(name) {
  const el = document.getElementById('file-name-display');
  el.textContent = `📹 ${name}`;
  el.style.display = 'inline-block';
}

// Analysis
async function analyseVideo() {
  const view      = document.querySelector('input[name="view"]:checked').value;
  const statusBar = document.getElementById('status-bar');
  const statusMsg = document.getElementById('status-msg');
  const errBanner = document.getElementById('error-banner');
  const btn       = document.getElementById('analyse-btn');

  if (!fileInput.files.length) {
    showError('Please select a video file first.');
    return;
  }

  // Clean up previous job frames
  if (jobId) {
    navigator.sendBeacon(`/cleanup/${jobId}`);
    jobId = null;
  }

  // Reset UI
  errBanner.style.display   = 'none';
  btn.disabled              = true;
  statusMsg.textContent     = 'Uploading and analysing — this may take a moment…';
  statusBar.style.display   = 'flex';
  document.getElementById('results').style.display = 'none';

  const formData = new FormData();
  formData.append('video', fileInput.files[0]);
  formData.append('view', view);

  try {
    const response = await fetch('/analyse', { method: 'POST', body: formData });
    const data     = await response.json();

    if (data.error) {
      showError(`Error: ${data.error}`);
      return;
    }

    currentReps   = data.reps;
    currentRepIdx = 0;
    jobId         = data.job_id;
    frameCount    = data.frame_count || 0;

    statusBar.style.display = 'none';
    frameCache.clear();
    currentFrame = 0;
    stopPlayback();

    renderResults();

  } catch (err) {
    showError('Something went wrong. Please try again.');
  } finally {
    btn.disabled            = false;
    statusBar.style.display = 'none';
  }
}

function showError(msg) {
  const el = document.getElementById('error-banner');
  el.textContent    = msg;
  el.style.display  = 'block';
  document.getElementById('status-bar').style.display = 'none';
}

// Render results 
function renderResults() {
  const repData = currentReps[currentRepIdx];

  // Rep tabs
  document.getElementById('rep-tabs').innerHTML = currentReps.map((_, i) =>
    `<button class="rep-tab ${i === currentRepIdx ? 'active' : ''}"
      onclick="selectRep(${i})">Rep ${i + 1}</button>`
  ).join('');

  // Summary counts
  const labels  = Object.values(repData.labels);
  const good    = labels.filter(l => l.severity === 'good').length;
  const warn    = labels.filter(l => l.severity === 'warn').length;
  const bad     = labels.filter(l => l.severity === 'bad').length;
  const overall      = bad > 0 ? 'Needs Work' : warn > 0 ? 'Acceptable' : 'Good';
  const overallClass = bad > 0 ? 'val-bad'    : warn > 0 ? 'val-warn'   : 'val-good';

  document.getElementById('summary-bar').innerHTML = `
    <div class="sum-item">
      <span class="sum-val ${overallClass}">${overall}</span>
      <span class="sum-lbl">Overall</span>
    </div>
    <div class="sum-item">
      <span class="sum-val val-good">${good}</span>
      <span class="sum-lbl">Correct</span>
    </div>
    <div class="sum-item">
      <span class="sum-val val-warn">${warn}</span>
      <span class="sum-lbl">Minor issues</span>
    </div>
    <div class="sum-item">
      <span class="sum-val val-bad">${bad}</span>
      <span class="sum-lbl">Needs attention</span>
    </div>
  `;

  // Feedback cards
  document.getElementById('feedback-cards').innerHTML =
    Object.entries(repData.labels).map(([key, fb]) => `
      <div class="fb-card">
        <div class="fb-label">${formatLabel(key)}</div>
        <div class="fb-card-header">
          <div class="fb-badge badge-${fb.severity}"></div>
          <div class="fb-status">${fb.status}</div>
        </div>
        <div class="fb-detail">${fb.detail}</div>
        ${fb.cue
          ? `<div class="cue-box"><p>${fb.cue}</p></div>`
          : `<p class="no-issue">No correction needed.</p>`}
      </div>`
    ).join('');

  // Scrubber
  if (jobId && frameCount > 0) {
    const slider = document.getElementById('frame-slider');
    slider.max   = frameCount - 1;
    slider.value = 0;
    document.getElementById('scrubber-panel').style.display = 'block';
    seekToFrame(0);
  } else {
    document.getElementById('scrubber-panel').style.display = 'none';
  }

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selectRep(idx) {
  currentRepIdx = idx;
  renderResults();
}

function formatLabel(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Frame scrubber

function frameUrl(idx) {
  return `/frames/${jobId}/${idx}`;
}

function preloadAround(idx) {
  // Preload ahead
  for (let i = idx; i < Math.min(idx + CACHE_AHEAD, frameCount); i++) {
    _cacheFrame(i);
  }
  // Keep a few behind too (for scrubbing backwards)
  for (let i = Math.max(0, idx - CACHE_BEHIND); i < idx; i++) {
    _cacheFrame(i);
  }
  // Evict frames far from current position
  for (const key of frameCache.keys()) {
    if (Math.abs(key - idx) > CACHE_AHEAD + CACHE_BEHIND + 20) {
      frameCache.delete(key);
    }
  }
}

function _cacheFrame(i) {
  if (frameCache.has(i)) return;
  const img = new Image();
  img.src = frameUrl(i);
  frameCache.set(i, img);
}

function seekToFrame(idx) {
  currentFrame = Math.max(0, Math.min(idx, frameCount - 1));

  const scrubberImg = document.getElementById('scrubber-img');
  const cached      = frameCache.get(currentFrame);

  if (cached && cached.complete && cached.naturalWidth > 0) {
    scrubberImg.src = cached.src;
  } else {
    scrubberImg.src = frameUrl(currentFrame);
  }

  document.getElementById('frame-slider').value   = currentFrame;
  document.getElementById('frame-badge').textContent =
    `${currentFrame + 1} / ${frameCount}`;

  preloadAround(currentFrame);
}

function stepFrame(delta) {
  seekToFrame(currentFrame + delta);
}

function togglePlayback() {
  if (playbackInterval) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  const playIcon  = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  playIcon.style.display  = 'none';
  pauseIcon.style.display = '';

  playbackInterval = setInterval(() => {
    const next = currentFrame + 1 >= frameCount ? 0 : currentFrame + 1;
    seekToFrame(next);
  }, 1000 / playbackFps);
}

function stopPlayback() {
  if (playbackInterval) {
    clearInterval(playbackInterval);
    playbackInterval = null;
  }
  const playIcon  = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');
  if (playIcon)  playIcon.style.display  = '';
  if (pauseIcon) pauseIcon.style.display = 'none';
}

function updatePlaybackSpeed() {
  playbackFps = parseInt(document.getElementById('speed-select').value, 10);
  if (playbackInterval) {
    stopPlayback();
    startPlayback();
  }
}

// Keyboard shortcuts for the scrubber
document.addEventListener('keydown', e => {
  if (!jobId || frameCount === 0) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(1); }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); stepFrame(-1); }
  if (e.key === ' ')          { e.preventDefault(); togglePlayback(); }
});

// Clean up server-side frames when the page is closed
window.addEventListener('beforeunload', () => {
  if (jobId) navigator.sendBeacon(`/cleanup/${jobId}`, '');
});