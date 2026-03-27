/* ── Grid Planning Page ─────────────────────────────────── */
const GridPage = {
  growthRates: { low: 0.04, base: 0.065, high: 0.09 },

  template() {
    return `
    <div class="page-header fade-up">
      <h1>⚡ Grid Planning</h1>
      <p>Capacity gap, renewable integration, demand scenarios & load analysis</p>
    </div>

    <!-- Capacity Summary KPIs -->
    <div class="kpi-grid fade-up" id="grid-kpi-row" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr))"></div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">Installed Capacity Breakdown</span></div>
        <div class="chart-wrap"><canvas id="chart-cap-breakdown"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Seasonal Demand Patterns</span></div>
        <div class="chart-wrap"><canvas id="chart-seasonal"></canvas></div>
      </div>
    </div>

    <div class="card fade-up chart-full">
      <div class="card-header"><span class="card-title">Capacity Gap Analysis — Daily Peak vs Installed Capacity</span></div>
      <div class="chart-wrap xlarge"><canvas id="chart-cap-gap"></canvas></div>
    </div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">🌱 Renewable Generation Trend</span></div>
        <div class="chart-wrap tall"><canvas id="chart-renewables"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Load Duration Curve</span></div>
        <div class="chart-wrap tall"><canvas id="chart-load-curve"></canvas></div>
      </div>
    </div>

    <div class="card fade-up chart-full">
      <div class="card-header">
        <span class="card-title">📊 Demand Growth Scenarios (2024–2035)</span>
      </div>
      <div style="padding:0 4px 20px">
        <div class="slider-row">
          <label>Conservative Growth Rate</label>
          <input type="range" id="sl-low" min="1" max="8" step="0.5" value="4" />
          <span class="slider-val" id="sl-low-val">4.0%</span>
        </div>
        <div class="slider-row">
          <label>Base Case Growth Rate</label>
          <input type="range" id="sl-base" min="3" max="12" step="0.5" value="6.5" />
          <span class="slider-val" id="sl-base-val">6.5%</span>
        </div>
        <div class="slider-row">
          <label>High Growth Rate</label>
          <input type="range" id="sl-high" min="6" max="15" step="0.5" value="9" />
          <span class="slider-val" id="sl-high-val">9.0%</span>
        </div>
      </div>
      <div class="chart-wrap xlarge"><canvas id="chart-scenarios"></canvas></div>
    </div>

    <div class="card fade-up chart-full">
      <div class="card-header"><span class="card-title">Peak Demand Heatmap (Avg by Hour × Day)</span></div>
      <div class="chart-wrap xlarge"><canvas id="chart-peak-heat"></canvas></div>
    </div>`;
  },

  async init() {
    document.getElementById('page-content').innerHTML = this.template();
    this.bindSliders();
    await Promise.all([
      this.loadCapacitySummary(),
      this.loadCapGap(),
      this.loadRenewables(),
      this.loadLoadCurve(),
      this.loadScenarios(),
      this.loadSeasonal(),
      this.loadPeakHeat(),
    ]);
  },

  async loadCapacitySummary() {
    const d = await API.capacitySummary();
    if (!d) return;
    document.getElementById('grid-kpi-row').innerHTML = [
      kpiCard({ id:'g-total', label:'Total Installed', value: d.total_installed_MW, unit:'MW', icon:'🏭', accent:'#6366f1' }),
      kpiCard({ id:'g-ren',   label:'Renewable Cap',  value: d.renewable_MW,        unit:'MW', icon:'🌱', accent:'#10b981' }),
      kpiCard({ id:'g-solar', label:'Solar Capacity', value: d.solar_MW,            unit:'MW', icon:'☀️', accent:'#f59e0b' }),
      kpiCard({ id:'g-wind',  label:'Wind Capacity',  value: d.wind_MW,             unit:'MW', icon:'💨', accent:'#8b5cf6' }),
      kpiCard({ id:'g-rpct',  label:'Renewable %',    value: d.renewable_pct,       unit:'%',  icon:'⚡', accent:'#06b6d4' }),
    ].join('');
    animateCounter(document.getElementById('g-total'), d.total_installed_MW, 900);
    animateCounter(document.getElementById('g-ren'),   d.renewable_MW, 900);
    animateCounter(document.getElementById('g-solar'), d.solar_MW, 900);
    animateCounter(document.getElementById('g-wind'),  d.wind_MW, 900);
    animateCounter(document.getElementById('g-rpct'),  d.renewable_pct, 900, 1);

    // Capacity breakdown donut
    destroyChart('chart-cap-breakdown');
    const ctx = document.getElementById('chart-cap-breakdown').getContext('2d');
    new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Thermal', 'Hydro', 'Solar', 'Wind', 'Other Ren.', 'Nuclear'],
        datasets: [{
          data: [d.thermal_MW, d.hydro_MW, d.solar_MW, d.wind_MW, 12000, d.nuclear_MW],
          backgroundColor: ['#ef4444','#06b6d4','#f59e0b','#8b5cf6','#10b981','#6366f1'],
          borderColor: '#111827', borderWidth: 3, hoverOffset: 10,
        }],
      },
      options: { ...chartOpts(), cutout: '58%' },
    });
  },

  async loadSeasonal() {
    const d = await API.seasonal();
    if (!d) return;
    destroyChart('chart-seasonal');
    const ctx = document.getElementById('chart-seasonal').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: d.seasons,
        datasets: [
          { label: 'Avg MW',  data: d.avg,  backgroundColor: 'rgba(99,102,241,0.75)', borderRadius: 6 },
          { label: 'Peak MW', data: d.peak, backgroundColor: 'rgba(245,158,11,0.7)',  borderRadius: 6 },
          { label: 'Min MW',  data: d.min,  backgroundColor: 'rgba(16,185,129,0.6)',  borderRadius: 6 },
        ],
      },
      options: chartOpts(),
    });
  },

  async loadCapGap() {
    const d = await API.capacityGap();
    if (!d) return;
    destroyChart('chart-cap-gap');
    const ctx = document.getElementById('chart-cap-gap').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.dates,
        datasets: [
          { label:'Installed Capacity', data: d.installed_capacity, borderColor:'#10b981', borderWidth:2, fill:false, tension:0.2, pointRadius:0 },
          { label:'Peak Demand',        data: d.peak_demand,        borderColor:'#ef4444', borderWidth:2, fill:false, tension:0.2, pointRadius:0 },
        ],
      },
      options: chartOpts({
        plugins: { legend: { display: true } },
        scales: { y: { ticks: { callback: v => fmt.number(v) + ' MW' } } },
      }),
    });
  },

  async loadRenewables() {
    const d = await API.renewables();
    if (!d) return;
    destroyChart('chart-renewables');
    const ctx = document.getElementById('chart-renewables').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: d.dates,
        datasets: [
          { label:'Solar MW', data: d.solar, backgroundColor:'rgba(245,158,11,0.75)', borderRadius:4, stack:'ren' },
          { label:'Wind MW',  data: d.wind,  backgroundColor:'rgba(139,92,246,0.75)', borderRadius:4, stack:'ren' },
        ],
      },
      options: chartOpts({ scales: { x: { stacked: true }, y: { stacked: true } } }),
    });
  },

  async loadLoadCurve() {
    const d = await API.loadCurve();
    if (!d) return;
    destroyChart('chart-load-curve');
    const ctx = document.getElementById('chart-load-curve').getContext('2d');
    const grad = ctx.createLinearGradient(0,0,0,300);
    grad.addColorStop(0,'rgba(99,102,241,0.4)');
    grad.addColorStop(1,'rgba(99,102,241,0.0)');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.percentile,
        datasets: [{
          label:'Demand MW', data: d.demand,
          borderColor:'#6366f1', backgroundColor: grad,
          borderWidth:2, fill:true, tension:0, pointRadius:0,
        }],
      },
      options: chartOpts({
        plugins: {legend:{display:false}},
        scales: {
          x: { title:{ display:true, text:'% of Hours', color:'#64748b' }, ticks:{maxTicksLimit:10} },
          y: { title:{ display:true, text:'Demand (MW)', color:'#64748b' } },
        },
      }),
    });
  },

  async loadScenarios(l, b, h) {
    const d = await API.scenarios(
      l ?? this.growthRates.low,
      b ?? this.growthRates.base,
      h ?? this.growthRates.high
    );
    if (!d) return;
    destroyChart('chart-scenarios');
    const ctx = document.getElementById('chart-scenarios').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: d.scenarios[0]?.years,
        datasets: d.scenarios.map(s => ({
          label: s.name, data: s.demand,
          borderColor: s.color, backgroundColor: s.color + '20',
          borderWidth: 2.5, fill: false, tension: 0.25, pointRadius: 4,
        })),
      },
      options: chartOpts({
        plugins: { legend: { display: true } },
        scales: { y: { ticks: { callback: v => fmt.number(v) + ' MW' } } },
      }),
    });
  },

  async loadPeakHeat() {
    const d = await API.peakHeatmap();
    if (!d) return;
    const days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    const mn = Math.min(...d.demand), mx = Math.max(...d.demand);
    const points = d.hour.map((h, i) => ({
      x: d.day[i], y: h,
      v: d.demand[i],
      norm: (d.demand[i] - mn) / (mx - mn),
    }));
    destroyChart('chart-peak-heat');
    const ctx = document.getElementById('chart-peak-heat').getContext('2d');
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
          pointStyle: 'rect', pointRadius: 18,
        }],
      },
      options: chartOpts({
        plugins: { legend:{display:false}, tooltip:{ callbacks:{
          label: ctx => `${days[ctx.raw.x]} ${ctx.raw.y}:00 — ${fmt.number(points[ctx.dataIndex].v)} MW`
        }}},
        scales: {
          x: { type:'linear', min:-0.5, max:6.5, ticks:{callback:v=>days[v]??'',color:'#64748b'} },
          y: { type:'linear', min:-0.5, max:23.5, ticks:{callback:v=>`${v}:00`,stepSize:3,color:'#64748b'} },
        },
      }),
    });
  },

  bindSliders() {
    const bind = (id, valId, key) => {
      const el = document.getElementById(id);
      const vEl = document.getElementById(valId);
      el.addEventListener('input', () => {
        const v = parseFloat(el.value);
        vEl.textContent = v.toFixed(1) + '%';
        this.growthRates[key] = v / 100;
        this.loadScenarios(this.growthRates.low, this.growthRates.base, this.growthRates.high);
      });
    };
    bind('sl-low','sl-low-val','low');
    bind('sl-base','sl-base-val','base');
    bind('sl-high','sl-high-val','high');
  },
};
