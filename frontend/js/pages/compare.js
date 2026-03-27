/* ── Model Comparison Page ──────────────────────────────── */
const MODEL_COLORS = {
  XGBoost:       '#6366f1',
  LightGBM:      '#10b981',
  RandomForest:  '#f59e0b',
};

const ComparePage = {
  template() {
    return `
    <div class="page-header fade-up">
      <h1>🤖 Model Comparison</h1>
      <p>Side-by-side evaluation of all trained forecasting models</p>
    </div>

    <div id="best-model-banner" class="fade-up"></div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">Performance Metrics</span></div>
        <div style="overflow-x:auto">
          <table class="data-table" id="metrics-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE (MW)</th>
                <th>RMSE (MW)</th>
                <th>MAPE (%)</th>
                <th>R² Score</th>
                <th>Train Time</th>
              </tr>
            </thead>
            <tbody id="metrics-body">
              <tr><td colspan="6"><div class="skeleton" style="height:40px"></div></td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Radar Chart — Overall Performance</span></div>
        <div class="chart-wrap tall"><canvas id="chart-radar"></canvas></div>
      </div>
    </div>

    <div class="card fade-up chart-full">
      <div class="card-header">
        <span class="card-title">Predictions vs Actual — All Models (First 168 Hours)</span>
      </div>
      <div id="compare-chart-area">
        <div class="empty-state">
          <div class="empty-icon">⏳</div>
          <p>Loading model predictions...</p>
        </div>
      </div>
    </div>

    <div class="chart-grid-2 fade-up">
      <div class="card">
        <div class="card-header"><span class="card-title">MAE Comparison</span></div>
        <div class="chart-wrap"><canvas id="chart-mae-bar"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">R² Score Comparison</span></div>
        <div class="chart-wrap"><canvas id="chart-r2-bar"></canvas></div>
      </div>
    </div>`;
  },

  async init() {
    document.getElementById('page-content').innerHTML = this.template();
    await Promise.all([
      this.loadMetrics(),
      this.loadPredictions(),
      this.loadRadar(),
    ]);
  },

  async loadMetrics() {
    const d = await API.metrics();
    if (!d || d.error) {
      document.getElementById('metrics-body').innerHTML =
        `<tr><td colspan="6"><div class="error-state" style="padding:24px"><div class="error-icon">🤖</div><p>${d?.error || 'Run training first'}</p></div></td></tr>`;
      return;
    }

    // Best model banner
    if (d.best_model) {
      document.getElementById('best-model-banner').innerHTML = `
        <div class="best-model-card">
          <div class="best-model-trophy">🏆</div>
          <div class="best-model-info">
            <h3>${d.best_model}</h3>
            <p>Best performing model based on lowest MAPE score</p>
            <div style="display:flex;gap:8px;margin-top:8px">
              ${d.models.filter(m=>m.name===d.best_model).map(m=>`
                <span class="chip chip-indigo">MAE ${fmt.number(m.mae)} MW</span>
                <span class="chip chip-green">MAPE ${m.mape?.toFixed(2)}%</span>
                <span class="chip chip-amber">R² ${m.r2?.toFixed(4)}</span>
              `).join('')}
            </div>
          </div>
        </div>`;
    }

    // Table
    const badgeClass = (name) => name === d.best_model ? 'badge badge-indigo' : 'badge badge-amber';
    document.getElementById('metrics-body').innerHTML = d.models.map(m => `
      <tr>
        <td>
          <span style="display:flex;align-items:center;gap:8px">
            <span style="width:10px;height:10px;border-radius:50%;background:${MODEL_COLORS[m.name]||'#6366f1'};display:inline-block"></span>
            <strong>${m.name}</strong>
            ${m.name === d.best_model ? '<span class="badge badge-indigo">Best</span>' : ''}
          </span>
        </td>
        <td class="num">${fmt.number(m.mae)}</td>
        <td class="num">${fmt.number(m.rmse)}</td>
        <td class="num">${m.mape?.toFixed(2)}</td>
        <td class="num">${m.r2?.toFixed(4)}</td>
        <td><span class="chip chip-green">${m.training_time_s}s</span></td>
      </tr>`).join('');

    // MAE and R2 bar charts
    const names = d.models.map(m => m.name);
    const colors = names.map(n => MODEL_COLORS[n] || '#6366f1');

    destroyChart('chart-mae-bar');
    new Chart(document.getElementById('chart-mae-bar').getContext('2d'), {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{ label:'MAE (MW)', data: d.models.map(m=>m.mae), backgroundColor: colors.map(c=>c+'bb'), borderRadius:8 }],
      },
      options: chartOpts({ plugins:{legend:{display:false}}, scales:{y:{title:{display:true,text:'MAE (MW)',color:'#64748b'}}} }),
    });

    destroyChart('chart-r2-bar');
    new Chart(document.getElementById('chart-r2-bar').getContext('2d'), {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{ label:'R² Score', data: d.models.map(m=>m.r2), backgroundColor: colors.map(c=>c+'bb'), borderRadius:8 }],
      },
      options: chartOpts({
        plugins:{legend:{display:false}},
        scales:{ y:{ min:0.9, max:1, title:{display:true,text:'R² Score',color:'#64748b'} } }
      }),
    });
  },

  async loadPredictions() {
    const d = await API.predictions(168);
    if (!d || d.error) {
      document.getElementById('compare-chart-area').innerHTML =
        `<div class="error-state"><p>${d?.error || 'No predictions available'}</p></div>`;
      return;
    }
    document.getElementById('compare-chart-area').innerHTML =
      `<div class="chart-wrap xlarge"><canvas id="chart-compare-preds"></canvas></div>`;
    destroyChart('chart-compare-preds');
    const ctx = document.getElementById('chart-compare-preds').getContext('2d');
    const labels = d.timestamps.map(t => t.slice(5,16).replace('T',' '));
    const modelKeys = Object.keys(d).filter(k => !['timestamps','actual'].includes(k));
    new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label:'Actual', data: d.actual, borderColor:'#94a3b8', borderWidth:1.5, fill:false, tension:0.2, pointRadius:0 },
          ...modelKeys.map(k => ({
            label: k, data: d[k],
            borderColor: MODEL_COLORS[k] || '#6366f1',
            borderWidth: 2, fill: false, tension: 0.2, pointRadius: 0,
          })),
        ],
      },
      options: chartOpts({ plugins:{ legend:{display:true} } }),
    });
  },

  async loadRadar() {
    const d = await API.radar();
    if (!d || !d.models.length) return;
    destroyChart('chart-radar');
    const ctx = document.getElementById('chart-radar').getContext('2d');
    new Chart(ctx, {
      type: 'radar',
      data: {
        labels: d.metric_labels,
        datasets: d.data.map(m => ({
          label: m.model,
          data: m.values,
          borderColor: MODEL_COLORS[m.model] || '#6366f1',
          backgroundColor: (MODEL_COLORS[m.model] || '#6366f1') + '25',
          borderWidth: 2, pointRadius: 4,
        })),
      },
      options: {
        ...chartOpts(),
        scales: {
          r: {
            min: 0, max: 1,
            ticks: { color:'#64748b', backdropColor:'transparent', stepSize:0.25 },
            grid:  { color:'rgba(99,102,241,0.1)' },
            pointLabels: { color:'#94a3b8', font:{ size:11 } },
          },
        },
      },
    });
  },
};
