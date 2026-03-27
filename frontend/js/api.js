/* ── API base URL (auto-detect) ──────────────────────────── */
const API_BASE = window.location.origin;

async function apiFetch(path, params = {}) {
  const url = new URL(API_BASE + path);
  Object.entries(params).forEach(([k, v]) => v != null && url.searchParams.set(k, v));
  try {
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`API error [${path}]:`, err);
    return null;
  }
}

/* ── Endpoint wrappers ───────────────────────────────────── */
const API = {
  // Overview
  kpis:        ()             => apiFetch('/api/overview/kpis'),
  trend:       (freq, start, end) => apiFetch('/api/overview/trend', { freq, start, end }),
  heatmap:     ()             => apiFetch('/api/overview/heatmap'),
  yearlyStats: ()             => apiFetch('/api/overview/yearly-stats'),

  // Regional
  breakdown:   ()             => apiFetch('/api/regional/breakdown'),
  regionTrend: (region, freq) => apiFetch('/api/regional/trend', { region, freq }),
  regionCompare:(freq)        => apiFetch('/api/regional/compare', { freq }),
  regionStats: ()             => apiFetch('/api/regional/stats'),

  // Forecast
  actuals:     (freq, start, end) => apiFetch('/api/forecast/actuals', { freq, start, end }),
  predict:     (model, steps)     => apiFetch('/api/forecast/predict', { model, steps }),
  featureImp:  (model)            => apiFetch('/api/forecast/feature-importance', { model }),

  // Grid
  capacitySummary: ()          => apiFetch('/api/grid/capacity-summary'),
  capacityGap:     ()          => apiFetch('/api/grid/capacity-gap'),
  renewables:      ()          => apiFetch('/api/grid/renewables'),
  scenarios:       (l, b, h)   => apiFetch('/api/grid/scenarios', { growth_low: l, growth_base: b, growth_high: h }),
  loadCurve:       ()          => apiFetch('/api/grid/load-curve'),
  peakHeatmap:     ()          => apiFetch('/api/grid/peak-heatmap'),
  seasonal:        ()          => apiFetch('/api/grid/seasonal-analysis'),

  // Compare
  metrics:     ()             => apiFetch('/api/compare/metrics'),
  predictions: (limit)        => apiFetch('/api/compare/predictions', { limit }),
  radar:       ()             => apiFetch('/api/compare/radar'),

  // Health
  health:      ()             => apiFetch('/api/health'),
};
