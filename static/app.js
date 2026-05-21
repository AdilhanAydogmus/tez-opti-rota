/* ═══════════════════════════════════════════════
   app.js — Shared utilities for all pages
═══════════════════════════════════════════════ */

// ── AUTH HELPERS ─────────────────────────────
const Auth = {
  getToken() { return localStorage.getItem('access_token'); },
  getUsername() { return localStorage.getItem('username'); },
  isLoggedIn() { return !!this.getToken(); },
  save(token, username) {
    localStorage.setItem('access_token', token);
    localStorage.setItem('username', username);
  },
  clear() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
  },
  headers() {
    const t = this.getToken();
    return t ? { 'Authorization': `Bearer ${t}` } : {};
  }
};

// ── TOAST ─────────────────────────────────────
const Toast = {
  show(msg, type='info', duration=4000) {
    let c = document.getElementById('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      document.body.appendChild(c);
    }
    const t = document.createElement('div');
    const icons = { success:'✅', error:'❌', warning:'⚠️', info:'💡' };
    t.className = `toast toast-${type}`;
    t.innerHTML = `<span>${icons[type]||'ℹ️'}</span><span>${msg}</span>`;
    c.appendChild(t);
    setTimeout(() => { t.style.opacity='0'; t.style.transform='translateX(100%)'; t.style.transition='all 0.3s'; setTimeout(()=>t.remove(),300); }, duration);
  }
};

// ── API FETCH ─────────────────────────────────
async function apiFetch(url, options={}) {
  const headers = { ...Auth.headers(), ...(options.headers||{}) };
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    Toast.show('Bu işlem için giriş yapmanız gerekiyor.', 'warning');
    throw new Error('UNAUTHORIZED');
  }
  return res;
}

// ── LOGIN GUARD ───────────────────────────────
function requireLogin(actionName='bu işlemi yapabilmek') {
  if (!Auth.isLoggedIn()) {
    Toast.show(`Lütfen giriş yapın — ${actionName} için oturum açmanız gerekiyor.`, 'warning', 5000);
    return false;
  }
  return true;
}

// ── NAVBAR ────────────────────────────────────
function initNavbar() {
  const userBadge = document.getElementById('user-badge');
  const loginBtn = document.getElementById('nav-login-btn');
  const logoutBtn = document.getElementById('nav-logout-btn');
  const hamburger = document.getElementById('hamburger');
  const navLinks = document.getElementById('nav-links');

  // Hamburger toggle
  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      hamburger.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', isOpen);
    });
    // Menü dışına tıklayınca kapat
    document.addEventListener('click', (e) => {
      if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
        navLinks.classList.remove('open');
        hamburger.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    });
    // Link tıklayınca kapat
    navLinks.querySelectorAll('.nav-link').forEach(l => {
      l.addEventListener('click', () => {
        navLinks.classList.remove('open');
        hamburger.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  function updateNav() {
    if (Auth.isLoggedIn()) {
      if (userBadge) { userBadge.style.display='flex'; userBadge.querySelector('.username').textContent = Auth.getUsername(); }
      if (loginBtn) loginBtn.style.display='none';
      if (logoutBtn) logoutBtn.style.display='flex';
    } else {
      if (userBadge) userBadge.style.display='none';
      if (loginBtn) loginBtn.style.display='flex';
      if (logoutBtn) logoutBtn.style.display='none';
    }
  }

  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      Auth.clear();
      Toast.show('Çıkış yapıldı.', 'info');
      setTimeout(() => location.href='/', 800);
    });
  }

  // active link
  const links = document.querySelectorAll('.nav-link');
  links.forEach(l => {
    if (l.getAttribute('href') === location.pathname) l.classList.add('active');
  });

  updateNav();
}

// ── TABS ─────────────────────────────────────
function initTabs(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  // Tab content'ler container'ın kardeşi veya page-wrap içinde olabilir
  // Bu yüzden document genelinde arama yapıyoruz
  const pageScope = container.closest('.page-wrap') || document;

  container.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      pageScope.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      const content = pageScope.querySelector(`[data-tab-content="${target}"]`);
      if (content) content.classList.add('active');
    });
  });
}

// ── FILE DROP ─────────────────────────────────
function initFileDrop(dropZoneId, inputId, labelId) {
  const zone = document.getElementById(dropZoneId);
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      if (label) label.textContent = e.dataTransfer.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    if (input.files[0] && label) label.textContent = input.files[0].name;
  });
}

// ── SESSION STORAGE FOR PIPELINE ─────────────
const Pipeline = {
  saveLSTMResult(data) { sessionStorage.setItem('lstm_result', JSON.stringify(data)); },
  getLSTMResult() { const d = sessionStorage.getItem('lstm_result'); return d ? JSON.parse(d) : null; },
  saveKumelemeResult(data) { sessionStorage.setItem('kumeleme_result', JSON.stringify(data)); },
  getKumelemeResult() { const d = sessionStorage.getItem('kumeleme_result'); return d ? JSON.parse(d) : null; },
};

// init on load
document.addEventListener('DOMContentLoaded', initNavbar);
