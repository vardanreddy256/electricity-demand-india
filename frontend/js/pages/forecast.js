/* ── Forecasting Page ──────────────────────────────────── */
const ForecastPage = {
  activeModel: 'xgboost',
  activeSteps: 168,

  template() {
    return `
    <div class="page-header fade-up">
      <h1>📈 Demand Forecasting</h1>
      <p>ML-powered electricity demand predictions with confidence intervals</p>
    </div>

    <div class="info-banner fade-up">
      ℹ️ Forecasts use the 2024 test period. Models trained on 2019–2022 data, validated on 2023.
    </div>

    <div class="controls-bar card fade-up" style="margin-bottom:24px;gap:20px">
      <div>
        <label style="display:block;margin-bottom:6px">Model</label>
        <select class="ctrl-select" id="model-select" style="min-width:160px">
          <option value="xgboost" selected>XGBoost</option>
          <option value="lightgbm">LightGBM</option>
          <option value="randomforest">Random Forest</option>
        </select>
      </div>
      <div>
        <label style="display:block;margin-bottom:6px">Forecast Horizon</label>
        <select class="ctrl-select" id="steps-select">
          <option value="24">24 hours (1 day)</option>
          <option value="72">72 hours (3 days)</option>
          <option value="168" selected>168 hours (1 week)</option>
          <option value="336">336 hours (2 weeks)</option>
          <option value="720">720 hours (1 month)</option>
        </select>
      </div>
      <div style="align-self:flex-end">
        <button class="btn active" id="run-forecast-btn" style="padding:10px 22px;font-size:13px">⚡ Run Forecast</button>
      </div>
    </div>

    <div class="metric-row fade-up" id="forecast-metrics">
      <div class="metric-box"><div class="m-label">MAE</div><div class="m-value" id="m-mae">—</div><div class="m-unit">MW</div></div>
      <div class="metric-box"><div class="m-label">RMSE</div><div class="m-value" id="m-rmse">—</div><div class="m-unit">MW</div></div>
      <div class="metric-box"><div class="m-label">MAPE</div><div class="m-value" id="m-mape">—</div><div class="m-unit">%</div></div>
      <div class="metric-box"><div class="m-label">R² Score</div><div class="m-value" id="m-r2">—</div><div class="m-unit"></div></div>
    </div>

    <div class="card fade-up chart-full">
      <div class="card-header">
        <span class="card-title">Actual vs Predicted Demand</span>
        <span class="chip chip-indigo" id="model-badge">XGBoost</span>
      </div>
      <div id="forecast-chart-area">
        <div class="empty-state">
          <div class="empty-icon">📈</div>
          <p>Select a model and click "Run Forecast" to see predictions</p>
        </div>
      </div>
    </div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">Feature Importance</span></div>
        <div class="chart-wrap tall"><canvas id="chart-feat-imp"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Prediction Error Distribution</span></div>
        <div class="chart-wrap tall"><canvas id="chart-error-dist"></canvas></div>
      </div>
    </div>`;
  },

  async init() {
    document.getElementById('page-content').innerHTML = this.template();
    document.getElementById('run-forecast-btn').addEventListener('click', () => this.runForecast());
    document.getElementById('model-select').addEventListener('change', e => {
      this.activeModel = e.target.value;
      document.getElementById('model-badge').textContent = e.target.options[e.target.selectedIndex].text;
    });
    document.getElementById('steps-select').addEventListener('change', e => {
      this.activeSteps = Number(e.target.value);
    });
    await this.runForecast();
  },

  async runForecast() {
    const btn = document.getElementById('run-forecast-btn');
    btn.textContent = '⏳ Running...';
    btn.disabled = true;
    const d = await API.predict(this.activeModel, this.activeSteps);
    btn.textContent = '⚡ Run Forecast';
    btn.disabled = false;

    if (!d || d.error) {
      document.getElementById('forecast-chart-area').innerHTML = `
        <div class="error-state"><div class="error-icon">🤖</div><h3>${d?.error || 'No data'}</h3>
        <p>Make sure training pipeline has run: <code>python scripts/run_pipeline.py</code></p></div>`;
      return;
    }

    // Metrics
    if (d.metrics) {
      document.getElementById('m-mae').textContent = fmt.number(d.metrics.mae);
      document.getElementById('m-rmse').textContent = fmt.number(d.metrics.rmse);
      document.getElementById('m-mape').textContent = d.metrics.mape?.toFixed(2);
      document.getElementById('m-r2').textContent = d.metrics.r2?.toFixed(4);
    }

    // Forecast chart
    document.getElementById('forecast-chart-area').innerHTML = `<div class="chart-wrap xlarge"><canvas id="chart-forecast"></canvas></div>`;
    destroyChart('chart-forecast');
    const ctx = document.getElementById('chart-forecast').getContext('2d');
    const labels = d.timestamps.map(t => t.slice(5,16).replace('T',' '));
    new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Actual', data: d.actual,
            borderColor: '#94a3b8', borderWidth: 1.5,
            fill: false, tension: 0.2, pointRadius: 0,
          },
          {
            label: 'Predicted', data: d.predicted,
            borderColor: '#6366f1', borderWidth: 2,
            fill: false, tension: 0.2, pointRadius: 0,
          },
          {
            label: 'Upper CI', data: d.upper_bound,
            borderColor: 'transparent', fill: '+1',
            backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.2, pointRadius: 0,
          },
          {
            label: 'Lower CI', data: d.lower_bound,
            borderColor: 'transparent', fill: false,
            backgroundColor: 'rgba(99,102,241,0.1)', tension: 0.2, pointRadius: 0,
          },
        ],
      },
      options: chartOpts({ plugins: { legend: { display: true } } }),
    });

    // Feature importance
    await this.loadFeatureImportance();

    // Error distribution
    const errors = d.actual.map((a, i) => Math.abs(a - d.predicted[i]));
    this.renderErrorDist(errors);
  },

  async loadFeatureImportance() {
    const d = await API.featureImp(this.activeModel);
    if (!d || !d.labels.length) return;
    destroyChart('chart-feat-imp');
    const ctx = document.getElementById('chart-feat-imp').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: d.labels,
        datasets: [{
          label: 'Importance',
          data: d.values,
          backgroundColor: d.labels.map((_, i) => `hsla(${240 + i*8},80%,65%,0.8)`),
          borderRadius: 4,
        }],
      },
      options: chartOpts({
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { font: { size: 10 } } },
          y: { ticks: { font: { size: 10 } } },
        },
      }),
    });
  },

  renderErrorDist(errors) {
    // Bin into 20 buckets
    const mx = Math.max(...errors);
    const buckets = 20;
    const width = mx / buckets;
    const counts = Array(buckets).fill(0);
    errors.forEach(e => { const b = Math.min(Math.floor(e / width), buckets - 1); counts[b]++; });
    const labels = counts.map((_, i) => `${fmt.number(i * width)}–${fmt.number((i+1)*width)}`);
    destroyChart('chart-error-dist');
    const ctx = document.getElementById('chart-error-dist').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{ label:'Count', data: counts, backgroundColor:'rgba(16,185,129,0.6)', borderRadius:4 }],
      },
      options: chartOpts({
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { maxRotation:45, font:{size:9} } } },
      }),
    });
  },
};
