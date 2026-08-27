'use strict';

// ── DOM refs ──
const textarea      = document.getElementById('article-input');
const charCountEl   = document.getElementById('char-count');
const submitBtn     = document.getElementById('submit-btn');
const btnLabel      = document.getElementById('btn-label');
const errorMsg      = document.getElementById('error-msg');
const verdictPlaceholder = document.getElementById('verdict-placeholder');
const verdictStamp  = document.getElementById('verdict-stamp');
const stampBadge    = document.getElementById('stamp-badge');
const stampSub      = document.getElementById('stamp-sub');
const confidenceRow = document.getElementById('confidence-row');
const confFill      = document.getElementById('conf-fill');
const confValue     = document.getElementById('conf-value');
const detailGrid    = document.getElementById('detail-grid');
const ledgerBody    = document.getElementById('ledger-body');
const clockEl       = document.getElementById('clock');

// ── Clock ──
function tickClock() {
  const now = new Date();
  clockEl.textContent = now.toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
tickClock();
setInterval(tickClock, 30000);

// ── Char counter ──
textarea.addEventListener('input', () => {
  charCountEl.textContent = textarea.value.length;
});

// ── Verdict config ──
const VERDICTS = {
  Credible:   { cls: 'credible',   badge: 'badge--credible',   fill: 'var(--green)', sub: 'Consistent with sourced, fact-based reporting' },
  Misleading: { cls: 'misleading', badge: 'badge--misleading', fill: 'var(--amber)', sub: 'May contain selective or framed information' },
  False:      { cls: 'false-r',    badge: 'badge--false',      fill: 'var(--brick)', sub: 'Consistent with fabricated or false news patterns' },
  Unverified: { cls: 'unverified', badge: 'badge--unverified', fill: 'var(--muted)', sub: 'Insufficient evidence to confirm or deny' },
};

function getV(prediction) {
  return VERDICTS[prediction] || VERDICTS['Unverified'];
}

// ── Render verdict ──
function renderVerdict(data) {
  const v = getV(data.prediction);

  // Stamp
  verdictPlaceholder.hidden = true;
  stampBadge.className = `stamp-badge ${v.cls}`;
  stampBadge.textContent = data.prediction.toUpperCase();
  stampSub.textContent = v.sub;
  verdictStamp.hidden = false;

  // Confidence bar
  confidenceRow.hidden = false;
  confFill.style.background = v.fill;
  confFill.style.boxShadow = `0 0 8px ${v.fill}80`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    confFill.style.width = `${data.confidence}%`;
  }));

  // Number counter
  animateCounter(confValue, 0, data.confidence, 900, '%');

  // Detail cards
  document.getElementById('detail-reason').textContent = data.reason || '—';
  document.getElementById('detail-credibility').textContent = data.source_credibility || '—';
  document.getElementById('detail-language').textContent = data.detected_language || '—';
  document.getElementById('detail-date').textContent = data.publication_date || '—';
  document.getElementById('detail-recommendation').textContent = data.recommendation || '—';
  document.getElementById('detail-ai-warn').textContent = data.ai_generated_warning || '—';
  detailGrid.hidden = false;
}

// ── Number counter animation ──
function animateCounter(el, from, to, duration, suffix = '') {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const val = from + (to - from) * easeOut(t);
    el.textContent = `${Math.round(val)}${suffix}`;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

// ── Main run prediction ──
async function runPrediction() {
  const text = textarea.value.trim();
  errorMsg.hidden = true;

  if (text.length < 10) {
    errorMsg.textContent = 'Please paste at least 10 characters of article text.';
    errorMsg.hidden = false;
    return;
  }

  // Loading state
  submitBtn.disabled = true;
  btnLabel.textContent = 'Analyzing...';
  submitBtn.classList.add('loading');

  // Reset result area
  verdictPlaceholder.hidden = false;
  verdictStamp.hidden = true;
  confidenceRow.hidden = true;
  detailGrid.hidden = true;
  confFill.style.width = '0%';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Something went wrong.');

    renderVerdict(data);
    await Promise.all([loadHistory(), loadStats()]);

  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.hidden = false;
  } finally {
    submitBtn.disabled = false;
    btnLabel.textContent = 'Run Verification';
    submitBtn.classList.remove('loading');
  }
}

// ── Load history ──
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const rows = await res.json();

    if (!rows.length) {
      ledgerBody.innerHTML = '<tr class="ledger-empty"><td colspan="5">No checks logged yet.</td></tr>';
      return;
    }

    ledgerBody.innerHTML = rows.map(r => {
      const v = getV(r.result);
      const title = escapeHtml(r.title || '').slice(0, 80);
      return `
        <tr data-id="${r.id}">
          <td class="excerpt">${title}${(r.title || '').length > 80 ? '…' : ''}</td>
          <td><span class="badge ${v.badge}">${r.result}</span></td>
          <td class="mono">${Number(r.confidence).toFixed(1)}%</td>
          <td class="mono">${formatTime(r.created_at)}</td>
          <td><button class="row-delete" title="Remove" data-id="${r.id}">✕</button></td>
        </tr>
      `;
    }).join('');
  } catch (_) { /* silent */ }
}

// ── Load stats ──
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const s = await res.json();

    const map = {
      'stat-total':      s.total,
      'stat-credible':   s.credible,
      'stat-misleading': s.misleading,
      'stat-false':      s.false,
      'stat-unverified': s.unverified,
      'stat-acc':        s.model_accuracy ? `${s.model_accuracy}%` : '—',
    };

    Object.entries(map).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = val;
      el.classList.remove('flash');
      void el.offsetWidth;
      el.classList.add('flash');
    });
  } catch (_) { /* silent */ }
}

// ── Delete history row ──
ledgerBody.addEventListener('click', async e => {
  if (!e.target.classList.contains('row-delete')) return;
  const id = e.target.dataset.id;
  await fetch(`/api/history/${id}`, { method: 'DELETE' });
  await Promise.all([loadHistory(), loadStats()]);
});

// ── Event listeners ──
submitBtn.addEventListener('click', runPrediction);
textarea.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') runPrediction();
});

// ── Helpers ──
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

// ── Init ──
loadHistory();
loadStats();
