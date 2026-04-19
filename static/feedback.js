let currentReps = [];
let currentRepIdx = 0;

async function analyseVideo() {
  const fileInput = document.getElementById("video-input");
  const view      = document.querySelector('input[name="view"]:checked').value;
  const statusMsg = document.getElementById("status-msg");
  const btn       = document.getElementById("analyse-btn");

  if (!fileInput.files.length) {
    statusMsg.textContent = "Please select a video file first.";
    return;
  }

  // Show loading state
  btn.disabled = true;
  statusMsg.textContent = "Uploading and analysing — this may take a moment...";
  document.getElementById("results").style.display = "none";

  const formData = new FormData();
  formData.append("video", fileInput.files[0]);
  formData.append("view", view);

  try {
    const response = await fetch("/analyse", { method: "POST", body: formData }); // calling api endpoint with form data 
    const data     = await response.json(); //the data var will now hold the form results

    if (data.error) {
      statusMsg.textContent = `Error: ${data.error}`;
      return;
    }

    currentReps    = data.reps;
    currentRepIdx  = 0;
    statusMsg.textContent = "";
    renderResults(); //calls the render results function (this is what leads to the rep data being displayed on the page)

  } catch (err) {
    statusMsg.textContent = "Something went wrong. Please try again.";
  } finally {
    btn.disabled = false;
  }
}

function renderResults() {
  const repData = currentReps[currentRepIdx];

  // Rep tabs
  const tabsEl = document.getElementById("rep-tabs");
  tabsEl.innerHTML = currentReps.map((_, i) =>
    `<button class="rep-tab ${i === currentRepIdx ? "active" : ""}"
      onclick="selectRep(${i})">Rep ${i + 1}</button>`
  ).join("");

  // Summary counts
  const labels = Object.values(repData.labels);
  const good   = labels.filter(l => l.severity === "good").length;
  const warn   = labels.filter(l => l.severity === "warn").length;
  const bad    = labels.filter(l => l.severity === "bad").length;
  const overall = bad > 0 ? "Needs work" : warn > 0 ? "Acceptable" : "Good";
  const overallClass = bad > 0 ? "val-bad" : warn > 0 ? "val-warn" : "val-good";

  document.getElementById("summary-bar").innerHTML = `
    <div class="sum-item"><span class="sum-val ${overallClass}">${overall}</span><span class="sum-lbl">Overall</span></div>
    <div class="sum-item"><span class="sum-val val-good">${good}</span><span class="sum-lbl">Correct</span></div>
    <div class="sum-item"><span class="sum-val val-warn">${warn}</span><span class="sum-lbl">Minor issues</span></div>
    <div class="sum-item"><span class="sum-val val-bad">${bad}</span><span class="sum-lbl">Needs attention</span></div>
  `;

  // Feedback cards
  document.getElementById("feedback-cards").innerHTML = Object.entries(repData.labels).map(
    ([key, fb]) => `
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
  ).join("");

  document.getElementById("results").style.display = "block";
}

function selectRep(idx) {
  currentRepIdx = idx;
  renderResults();
}

function formatLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}