/* ── Overview page ──────────────────────────────────────── */
const OverviewPage = {
  activePeriod: '1Y',

  template() {
    return `
    <div class="page-header fade-up">
      <h1>⚡ National Overview</h1>
      <p>India electricity demand intelligence — real-time KPIs, trends & patterns</p>
    </div>

    <div class="kpi-grid" id="overview-kpis">
      ${[1,2,3,4,5].map(i => `<div class="kpi-card skeleton" style="height:110px"></div>`).join('')}
    </div>

    <div class="chart-full card fade-up">
      <div class="card-header">
        <span class="card-title">National Demand Trend</span>
        <div class="btn-group" id="trend-period-btns">
          ${['1M','3M','6M','1Y','2Y','ALL'].map(p =>
            `<button class="btn ${p==='1Y'?'active':''}" data-period="${p}">${p}</button>`
          ).join('')}
        </div>
      </div>
      <div class="chart-wrap xlarge">
        <canvas id="chart-trend"></canvas>
      </div>
    </div>

    <div class="chart-grid-2">
      <div class="card fade-up">
        <div class="card-header"><span class="card-title">Yearly Demand Summary</span></div>
        <div class="chart-wrap tall"><canvas id="chart-yearly"></canvas></div>
      </div>
      <div class="card fade-up">
        <div class="card-header"><span class="card-title">Demand Heatmap (Avg by Hour × Day)</span></div>
        <div class="chart-wrap tall"><canvas id="chart-heatmap"></canvas></div>
      </div>
    </div>`;
  },

  async init() {
    document.getElementById('page-content').innerHTML = this.template();
    this.bindPeriodBtns();
    await Promise.all([this.loadKPIs(), this.loadTrend('1Y', 'W'), this.loadYearly(), this.loadHeatmap()]);
  },

  async loadKPIs() {
    const d = await API.kpis();
    if (!d) return;
    document.getElementById('overview-kpis').innerHTML = [
      kpiCard({ id:'kpi-peak', label:'Peak Demand', value: d.peak_demand_MW, unit:'MW', icon:'🔺', accent:'#ef4444' }),
      kpiCard({ id:'kpi-avg',  label:'Avg Demand',  value: d.avg_demand_MW,  unit:'MW', icon:'📊', accent:'#6366f1' }),
      kpiCard({ id:'kpi-lf',   label:'Load Factor', value: d.load_factor_pct, unit:'%', icon:'⚡', accent:'#10b981' }),
      kpiCard({ id:'kpi-ren',  label:'Renewable Share', value: d.renewable_share_pct, unit:'%', change: null, icon:'☀️', accent:'#f59e0b' }),
      kpiCard({ id:'kpi-yoy',  label:'YoY Growth', value: d.yoy_growth_pct, unit:'%', icon:'📈', accent:'#8b5cf6', change: null }),
    ].join('');
    // Animate counters
    animateCounter(document.getElementById('kpi-peak'), d.peak_demand_MW, 900);
    animateCounter(document.getElementById('kpi-avg'),  d.avg_demand_MW,  900);
    animateCounter(document.getElementById('kpi-lf'),   d.load_factor_pct,900, 1);
    animateCounter(document.getElementById('kpi-ren'),  d.renewable_share_pct,900,1);
    animateCounter(document.getElementById('kpi-yoy'),  d.yoy_growth_pct, 900, 2);
  },

  async loadTrend(period = '1Y', freq = 'W') {
    const { start, end } = getTrendDates(period);
    const params = period === 'ALL' ? { freq: 'ME' } : { freq, start, end };
    const d = await API.trend(params.freq, params.start, params.end);
    if (!d || !d.timestamps || d.timestamps.length === 0) return;
    destroyChart('chart-trend');
    const ctx = document.getElementById('chart-trend').getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 380);
    grad.addColorStop(0, 'rgba(99,102,241,0.35)');
    grad.addColorStop(1, 'rgba(99,102,241,0.0)');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.timestamps.map(t => t.slice(0,10)),
        datasets: [{
          label: 'National Demand (MW)',
          data: d.demand,
          borderColor: '#6366f1',
          backgroundColor: grad,
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHoverRadius: 5,
        }],
      },
      options: chartOpts({ plugins: { legend: { display: false } } }),
    });
  },

  async loadYearly() {
    const d = await API.yearlyStats();
    if (!d) return;
    destroyChart('chart-yearly');
    const ctx = document.getElementById('chart-yearly').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: d.years,
        datasets: [
          { label: 'Avg MW',  data: d.avg,  backgroundColor: 'rgba(99,102,241,0.75)', borderRadius: 6 },
          { label: 'Peak MW', data: d.peak, backgroundColor: 'rgba(239,68,68,0.6)',   borderRadius: 6 },
        ],
      },
      options: chartOpts({ scales: { x: { ticks: { color: '#64748b' } } } }),
    });
  },

  async loadHeatmap() {
    const d = await API.heatmap();
    if (!d) return;
    // Build a 24x7 matrix
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    const matrix = Array.from({ length: 24 }, () => Array(7).fill(0));
    const counts = Array.from({ length: 24 }, () => Array(7).fill(0));
    d.hour.forEach((h, i) => {
      const day = d.day[i];
      matrix[h][day] = (matrix[h][day] || 0) + d.demand[i];
      counts[h][day]++;
    });
    const flat = matrix.flat();
    const mn = Math.min(...flat), mx = Math.max(...flat);

    destroyChart('chart-heatmap');
    const ctx = document.getElementById('chart-heatmap').getContext('2d');
    // Represent as scatter with colored squares
    const points = [];
    matrix.forEach((row, h) => row.forEach((val, day) => {
      const norm = (val - mn) / (mx - mn);
      points.push({ x: day, y: h, v: val, norm });
    }));
    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          data: points.map(p => ({ x: p.x, y: p.y })),
          backgroundColor: points.map(p => {
            const r = Math.round(99 + (239-99)*p.norm);
            const g = Math.round(102 + (68-102)*p.norm);
            const b = Math.round(241 + (68-241)*p.norm);
            return `rgba(${r},${g},${b},0.85)`;
          }),
          pointStyle: 'rect',
          pointRadius: 16,
        }],
      },
      options: chartOpts({
        plugins: { legend: {display:false}, tooltip: { callbacks: {
          label: ctx => `Hour ${ctx.raw.y}h, ${days[ctx.raw.x]}: ${fmt.number(points[ctx.dataIndex].v)} MW`
        }}},
        scales: {
          x: { type:'linear', min:-0.5, max:6.5, ticks: { callback: v => days[v], color:'#64748b' } },
          y: { type:'linear', min:-0.5, max:23.5, ticks: { callback: v => `${v}:00`, color:'#64748b', stepSize:4 } },
        },
      }),
    });
  },

  bindPeriodBtns() {
    document.getElementById('trend-period-btns')?.addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-period]');
      if (!btn) return;
      document.querySelectorAll('#trend-period-btns .btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const p = btn.dataset.period;
      this.activePeriod = p;
      const freqMap = { '1M':'D', '3M':'D', '6M':'W', '1Y':'W', '2Y':'ME', 'ALL':'ME' };
      await this.loadTrend(p, freqMap[p] || 'W');
    });
  },
};
