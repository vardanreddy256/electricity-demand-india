/* ── SPA Router & App Controller ────────────────────────── */
const PAGES = {
  overview: { module: OverviewPage,  title: 'Overview' },
  regional: { module: RegionalPage,  title: 'Regional Analysis' },
  forecast: { module: ForecastPage,  title: 'Forecasting' },
  grid:     { module: GridPage,      title: 'Grid Planning' },
  compare:  { module: ComparePage,   title: 'Model Comparison' },
};

let currentPage = null;

async function navigateTo(pageId) {
  if (currentPage === pageId) return;
  currentPage = pageId;

  // Update nav active state
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === pageId);
  });

  // Update page title
  document.getElementById('page-title').textContent = PAGES[pageId]?.title || pageId;

  // Clear & load page
  const content = document.getElementById('page-content');
  content.innerHTML = '';
  showLoading('Loading ' + (PAGES[pageId]?.title || pageId) + '...');

  try {
    await PAGES[pageId].module.init();
  } catch (err) {
    console.error('Page error:', err);
    content.innerHTML = `<div class="error-state"><div class="error-icon">⚠️</div><h3>Page Error</h3><p>${err.message}</p></div>`;
  } finally {
    hideLoading();
  }
}

// ── Sidebar toggle ────────────────────────────────────────
function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  const main = document.getElementById('main');
  const toggleBtn = document.getElementById('sidebar-toggle');

  let isMobile = () => window.innerWidth <= 900;

  toggleBtn.addEventListener('click', () => {
    if (isMobile()) {
      sidebar.classList.toggle('open');
    } else {
      sidebar.classList.toggle('collapsed');
      main.classList.toggle('expanded');
    }
  });

  // Close sidebar on mobile nav click
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', () => {
      if (isMobile()) sidebar.classList.remove('open');
    });
  });

  // Responsive check
  window.addEventListener('resize', () => {
    if (!isMobile()) sidebar.classList.remove('open');
  });
}

// ── Navigation binding ────────────────────────────────────
function initNav() {
  document.querySelectorAll('.nav-item[data-page]').forEach(el => {
    el.addEventListener('click', async (e) => {
      e.preventDefault();
      await navigateTo(el.dataset.page);
    });
  });
}

// ── Date display ──────────────────────────────────────────
function initDateDisplay() {
  const el = document.getElementById('current-date');
  const update = () => {
    const now = new Date();
    el.textContent = now.toLocaleString('en-IN', {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };
  update();
  setInterval(update, 60000);
}

// ── API health check ──────────────────────────────────────
async function checkHealth() {
  const h = await API.health();
  setApiStatus(h && h.status === 'ok');
}

// ── Bootstrap ─────────────────────────────────────────────
async function init() {
  initSidebar();
  initNav();
  initDateDisplay();
  await checkHealth();
  // Default page
  await navigateTo('overview');
  setInterval(checkHealth, 30000);
}

document.addEventListener('DOMContentLoaded', init);
