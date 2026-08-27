const textarea = document.getElementById("article-input");
const charCount = document.getElementById("char-count");
const submitBtn = document.getElementById("submit-btn");
const errorMsg = document.getElementById("error-msg");
const verdictStage = document.getElementById("verdict-stage");
const confidenceRow = document.getElementById("confidence-row");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceValue = document.getElementById("confidence-value");
const ledgerBody = document.getElementById("ledger-body");
const clockEl = document.getElementById("clock");

textarea.addEventListener("input", () => {
  charCount.textContent = textarea.value.length;
});

function tickClock() {
  const now = new Date();
  clockEl.textContent = now.toLocaleString(undefined, {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
tickClock();
setInterval(tickClock, 30000);

function renderStamp(prediction) {
  const isReal = prediction === "REAL";
  verdictStage.innerHTML = "";
  const stamp = document.createElement("div");
  stamp.className = "stamp";
  stamp.style.setProperty("--stamp-color", isReal ? "var(--gold)" : "var(--brick)");
  stamp.innerHTML = isReal
    ? "VERIFIED<span class='stamp__sub'>Consistent with real news patterns</span>"
    : "DISPUTED<span class='stamp__sub'>Consistent with fake news patterns</span>";
  verdictStage.appendChild(stamp);
}

function badge(prediction) {
  const isReal = prediction === "REAL";
  return `<span class="badge ${isReal ? "badge--real" : "badge--fake"}">${prediction}</span>`;
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

async function runPrediction() {
  const text = textarea.value.trim();
  errorMsg.hidden = true;

  if (text.length < 10) {
    errorMsg.textContent = "Please paste at least 10 characters of article text.";
    errorMsg.hidden = false;
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    renderStamp(data.prediction);
    confidenceRow.hidden = false;
    confidenceFill.style.width = `${data.confidence}%`;
    confidenceFill.style.background = data.prediction === "REAL" ? "var(--gold)" : "var(--brick)";
    confidenceValue.textContent = `${data.confidence}%`;

    await Promise.all([loadHistory(), loadStats()]);
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Run Verification";
  }
}

async function loadHistory() {
  const res = await fetch("/api/history");
  const rows = await res.json();

  if (!rows.length) {
    ledgerBody.innerHTML = `<tr class="ledger__empty"><td colspan="5">No checks logged yet.</td></tr>`;
    return;
  }

  ledgerBody.innerHTML = rows
    .map(
      (r) => `
      <tr data-id="${r.id}">
        <td class="excerpt">${escapeHtml(r.text).slice(0, 90)}${r.text.length > 90 ? "…" : ""}</td>
        <td>${badge(r.prediction)}</td>
        <td class="mono">${Math.round(r.confidence * 100)}%</td>
        <td class="mono">${formatTime(r.created_at)}</td>
        <td><button class="row-delete" title="Remove" data-id="${r.id}">✕</button></td>
      </tr>`
    )
    .join("");
}

async function loadStats() {
  const res = await fetch("/api/stats");
  const s = await res.json();
  document.getElementById("stat-total").textContent = s.total;
  document.getElementById("stat-real").textContent = s.real;
  document.getElementById("stat-fake").textContent = s.fake;
  document.getElementById("stat-conf").textContent = `${s.avg_confidence}%`;
  document.getElementById("stat-acc").textContent = s.model_accuracy ? `${s.model_accuracy}%` : "—";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

ledgerBody.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("row-delete")) return;
  const id = e.target.dataset.id;
  await fetch(`/api/history/${id}`, { method: "DELETE" });
  await Promise.all([loadHistory(), loadStats()]);
});

submitBtn.addEventListener("click", runPrediction);
textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runPrediction();
});

loadHistory();
loadStats();
