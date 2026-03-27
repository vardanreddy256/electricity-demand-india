/* ── Regional Analysis page ─────────────────────────────── */
const REGION_COLORS = {
  Northern: '#6366f1', Western: '#10b981',
  Eastern: '#f59e0b', Southern: '#ef4444', NorthEastern: '#8b5cf6',
};
const REGION_LABELS = { Northern:'Northern', Western:'Western', Eastern:'Eastern', Southern:'Southern', NorthEastern:'North-Eastern' };

const RegionalPage = {
  activeRegion: 'Northern',

  template() {
    return `
    <div class="page-header fade-up">
      <h1>🗺️ Regional Analysis</h1>
      <p>Electricity demand breakdown across India's 5 power grid regions</p>
    </div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">Demand Share by Region</span></div>
        <div class="chart-wrap"><canvas id="chart-region-donut"></canvas></div>
        <div id="region-legend" class="chart-legend-custom" style="margin-top:16px"></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Regional Stats Overview</span></div>
        <div style="overflow-x:auto">
          <table class="data-table" id="region-stats-table">
            <thead><tr><th>Region</th><th>Avg MW</th><th>Peak MW</th><th>Share %</th></tr></thead>
            <tbody id="region-stats-body"><tr><td colspan="4" class="skeleton" style="height:40px"></td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="card fade-up">
      <div class="card-header">
        <span class="card-title">Monthly Regional Trend</span>
        <div class="controls-bar">
          <label>Region:</label>
          <select class="ctrl-select" id="region-select">
            ${Object.entries(REGION_LABELS).map(([k,v]) =>
              `<option value="${k}" ${k===this.activeRegion?'selected':''}>${v}</option>`
            ).join('')}
          </select>
        </div>
      </div>
      <div class="chart-wrap xlarge"><canvas id="chart-region-trend"></canvas></div>
    </div>

    <div class="card fade-up">
      <div class="card-header"><span class="card-title">All Regions Monthly Comparison</span></div>
      <div class="chart-wrap xlarge"><canvas id="chart-region-compare"></canvas></div>
    </div>`;
  },

  async init() {
    document.getElementById('page-content').innerHTML = this.template();
    document.getElementById('region-select').addEventListener('change', async (e) => {
      this.activeRegion = e.target.value;
      await this.loadRegionTrend();
    });
    await Promise.all([
      this.loadBreakdown(),
      this.loadStats(),
      this.loadRegionTrend(),
      this.loadCompare(),
    ]);
  },

  async loadBreakdown() {
    const d = await API.breakdown();
    if (!d) return;
    destroyChart('chart-region-donut');
    const ctx = document.getElementById('chart-region-donut').getContext('2d');
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: d.regions.map(r => REGION_LABELS[r] || r),
        datasets: [{
          data: d.demand,
          backgroundColor: d.regions.map(r => REGION_COLORS[r] || '#6366f1'),
          borderColor: '#111827', borderWidth: 3,
          hoverOffset: 12,
        }],
      },
      options: {
        ...chartOpts({ plugins: { legend: { display:false } } }),
        cutout: '62%',
      },
    });
    // Custom legend
    document.getElementById('region-legend').innerHTML = d.regions.map((r, i) => `
      <div class="chart-legend-item">
        <div class="chart-legend-dot" style="background:${d.colors[i]}"></div>
        <span>${REGION_LABELS[r] || r}: <strong>${d.share[i]}%</strong></span>
      </div>`).join('');
  },

  async loadStats() {
    const d = await API.regionStats();
    if (!d) return;
    document.getElementById('region-stats-body').innerHTML = d.regions.map(r => `
      <tr>
        <td><span class="chip" style="background:${REGION_COLORS[r.region]}22;color:${REGION_COLORS[r.region]};border-color:${REGION_COLORS[r.region]}44">${REGION_LABELS[r.region]||r.region}</span></td>
        <td class="num">${fmt.number(r.avg_MW)}</td>
        <td class="num">${fmt.number(r.peak_MW)}</td>
        <td><div class="progress-bar-wrap" style="width:120px"><div class="progress-bar-fill" style="width:${r.share_pct}%;background:${REGION_COLORS[r.region]}"></div></div><span style="font-size:11px;color:var(--text-muted);margin-left:6px">${r.share_pct}%</span></td>
      </tr>`).join('');
  },

  async loadRegionTrend() {
    const d = await API.regionTrend(this.activeRegion, 'W');
    if (!d) return;
    destroyChart('chart-region-trend');
    const ctx = document.getElementById('chart-region-trend').getContext('2d');
    const color = REGION_COLORS[this.activeRegion] || '#6366f1';
    const grad = ctx.createLinearGradient(0,0,0,350);
    grad.addColorStop(0, color + 'aa');
    grad.addColorStop(1, color + '00');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.timestamps.map(t => t.slice(0,10)),
        datasets: [{
          label: `${REGION_LABELS[this.activeRegion]} Region (MW)`,
          data: d.demand,
          borderColor: color, backgroundColor: grad,
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 0,
        }],
      },
      options: chartOpts(),
    });
  },

  async loadCompare() {
    const d = await API.regionCompare('ME');
    if (!d || !d.dates) return;
    destroyChart('chart-region-compare');
    const ctx = document.getElementById('chart-region-compare').getContext('2d');
    const datasets = Object.keys(REGION_COLORS)
      .filter(r => d[r])
      .map(r => ({
        label: REGION_LABELS[r] || r,
        data: d[r],
        borderColor: REGION_COLORS[r],
        backgroundColor: REGION_COLORS[r] + '20',
        borderWidth: 2, fill: false, tension: 0.3, pointRadius: 1,
      }));
    new Chart(ctx, {
      type: 'line',
      data: { labels: d.dates, datasets },
      options: chartOpts(),
    });
  },
};
