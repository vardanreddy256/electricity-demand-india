/* ── Utility helpers ────────────────────────────────────── */
const fmt = {
  number: (v, dec = 0) => v == null ? '—' : Number(v).toLocaleString('en-IN', { minimumFractionDigits: dec, maximumFractionDigits: dec }),
  mw:  v => v == null ? '—' : `${fmt.number(v)} MW`,
  pct: v => v == null ? '—' : `${Number(v).toFixed(1)}%`,
  date: d => d ? new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year:'numeric' }) : '—',
};

function animateCounter(el, target, duration = 800, dec = 0) {
  const start = performance.now();
  const from = 0;
  const update = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = fmt.number(from + (target - from) * ease, dec);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function showLoading(msg = 'Loading...') {
  const el = document.getElementById('loading-overlay');
  el.querySelector('.spinner-text').textContent = msg;
  el.classList.add('visible');
}
function hideLoading() {
  document.getElementById('loading-overlay').classList.remove('visible');
}

function showError(containerId, msg) {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="error-state"><div class="error-icon">⚠️</div><h3>Data unavailable</h3><p>${msg}</p></div>`;
}

function setApiStatus(online) {
  const dot  = document.getElementById('api-status-dot');
  const text = document.getElementById('api-status-text');
  if (online) { dot.className = 'status-dot online'; text.textContent = 'API Online'; }
  else        { dot.className = 'status-dot offline'; text.textContent = 'API Offline'; }
}

function kpiCard({ id, label, value, unit, change, icon, accent = '#6366f1' }) {
  const changeHtml = change != null
    ? `<div class="kpi-change ${change >= 0 ? 'up' : 'down'}">${change >= 0 ? '↑' : '↓'} ${Math.abs(change).toFixed(1)}% YoY</div>`
    : '';
  return `
    <div class="kpi-card" style="--accent:${accent}">
      <div class="kpi-icon">${icon}</div>
      <div class="kpi-label">${label}</div>
      <div class="kpi-value" id="${id}">${fmt.number(value)}<span class="kpi-unit">${unit}</span></div>
      ${changeHtml}
    </div>`;
}

function destroyChart(id) {
  const existing = Chart.getChart(id);
  if (existing) existing.destroy();
}

// Default chart options
const chartDefaults = {
  responsive: true,
  maintainAspectRatio: true,
  animation: { duration: 600 },
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, boxWidth: 12, padding: 16 } },
    tooltip: {
      backgroundColor: '#1a2235',
      borderColor: 'rgba(99,102,241,0.3)',
      borderWidth: 1,
      titleColor: '#e2e8f0',
      bodyColor: '#94a3b8',
      padding: 12,
      cornerRadius: 8,
    },
  },
  scales: {
    x: {
      ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 11 } },
      grid:  { color: 'rgba(99,102,241,0.07)' },
    },
    y: {
      ticks: { color: '#64748b', font: { size: 11 } },
      grid:  { color: 'rgba(99,102,241,0.07)' },
    },
  },
};

function deepMerge(target, source) {
  for (const k in source) {
    if (source[k] && typeof source[k] === 'object' && !Array.isArray(source[k])) {
      target[k] = target[k] || {};
      deepMerge(target[k], source[k]);
    } else {
      target[k] = source[k];
    }
  }
  return target;
}

function chartOpts(overrides = {}) {
  return deepMerge(JSON.parse(JSON.stringify(chartDefaults)), overrides);
}

// Gradient helper for charts
function makeGradient(ctx, color, alpha1 = 0.35, alpha2 = 0.0) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, color.replace(')', `,${alpha1})`).replace('rgb', 'rgba'));
  gradient.addColorStop(1, color.replace(')', `,${alpha2})`).replace('rgb', 'rgba'));
  return gradient;
}

// Date range for trend filter — dataset ends 2024-04-30
function getTrendDates(period) {
  const DATASET_END = new Date('2024-04-30');
  const end = DATASET_END;
  const start = new Date(DATASET_END);
  const map = { '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'ALL': 9999 };
  start.setDate(start.getDate() - (map[period] || 365));
  return {
    start: start.toISOString().split('T')[0],
    end: end.toISOString().split('T')[0],
  };
}
