# ruff: noqa: E501
from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSeek Python API Dashboard</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 24px; background: linear-gradient(135deg, #1e3a8a, #312e81); }
    main { padding: 24px; max-width: 1100px; margin: 0 auto; }
    h1 { margin: 0 0 8px; }
    h2 { margin-top: 32px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
    .card, form, .toolbar { background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 16px; box-shadow: 0 8px 20px #02061755; }
    label { display: block; font-size: 13px; color: #94a3b8; margin: 10px 0 4px; }
    input, select { width: 100%; box-sizing: border-box; border-radius: 10px; border: 1px solid #475569; background: #020617; color: #e2e8f0; padding: 10px; }
    button { border: 0; border-radius: 10px; background: #2563eb; color: white; padding: 10px 12px; margin: 6px 6px 0 0; cursor: pointer; font-size: 13px; }
    button.secondary { background: #475569; }
    button.danger { background: #dc2626; }
    button.good { background: #16a34a; }
    button.warning { background: #d97706; }
    button.sm { padding: 4px 10px; font-size: 12px; margin: 0 4px 0 0; }
    .muted { color: #94a3b8; font-size: 13px; }
    .status { display: inline-block; border-radius: 999px; padding: 3px 9px; font-size: 12px; background: #475569; }
    .healthy { background: #166534; }
    .unhealthy, .disabled { background: #991b1b; }
    .cooldown { background: #92400e; }
    .tag { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; margin-right: 3px; }
    .tag-static   { background: #1e3a5f; color: #93c5fd; }
    .tag-rotating { background: #3b1a6e; color: #c4b5fd; }
    .tag-ready    { background: #14532d; color: #86efac; }
    .tag-cooldown { background: #78350f; color: #fcd34d; }
    .tag-enabled  { background: #14532d; color: #86efac; }
    .tag-disabled { background: #7f1d1d; color: #fca5a5; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #020617; padding: 12px; border-radius: 12px; color: #cbd5e1; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
    th { text-align: left; color: #94a3b8; padding: 6px 10px; font-weight: 500; border-bottom: 1px solid #334155; }
    td { padding: 7px 10px; border-bottom: 1px solid #1e293b; vertical-align: middle; }
    tr.disabled-row td { opacity: 0.5; }
  </style>
</head>
<body>
<header>
  <h1>DeepSeek Python API Dashboard</h1>
  <div class="muted">Manage DeepSeek web tokens, health checks, and rotation. Raw tokens are never displayed after submission.</div>
</header>
<main>
  <section class="toolbar">
    <label for="key">Management API key</label>
    <input id="key" type="password" placeholder="MANAGEMENT_API_KEY or API_KEY">
    <button onclick="saveKey()">Save key locally</button>
    <button class="secondary" onclick="loadAll()">Refresh</button>
    <button class="good" onclick="checkAll()">Check all tokens</button>
    <div class="muted">The key is stored only in this browser's localStorage.</div>
  </section>

  <h2>Status</h2>
  <div id="summary" class="grid"></div>

  <h2>Proxy Pool</h2>
  <div id="proxy-pool-section"></div>

  <h2>Rotation</h2>
  <form onsubmit="setRotation(event)">
    <label for="strategy">Strategy</label>
    <select id="strategy">
      <option value="fill_first">fill_first</option>
      <option value="round_robin">round_robin</option>
    </select>
    <button>Save strategy</button>
  </form>

  <h2>Add token</h2>
  <form onsubmit="addToken(event)">
    <label for="name">Name</label>
    <input id="name" placeholder="Personal account">
    <label for="token">DeepSeek web token</label>
    <input id="token" type="password" required>
    <label for="proxy">Proxy override (optional)</label>
    <input id="proxy" type="password" placeholder="Leave blank to assign from proxies.txt">
    <label><input id="check" type="checkbox" style="width:auto"> Check immediately</label>
    <button>Add token</button>
  </form>

  <h2>Tokens</h2>
  <div id="tokens" class="grid"></div>

  <h2>Log</h2>
  <pre id="log"></pre>
</main>
<script>
const keyInput = document.getElementById('key');
const strategyInput = document.getElementById('strategy');
const nameInput = document.getElementById('name');
const tokenInput = document.getElementById('token');
const checkInput = document.getElementById('check');
const proxyInput = document.getElementById('proxy');
const logOutput = document.getElementById('log');
const summaryOutput = document.getElementById('summary');
const tokensOutput = document.getElementById('tokens');
const proxyPoolSection = document.getElementById('proxy-pool-section');
keyInput.value = localStorage.getItem('deepseekManagementKey') || '';
function saveKey() { localStorage.setItem('deepseekManagementKey', keyInput.value); log('Saved key locally.'); }
function headers() { return { 'Authorization': `Bearer ${keyInput.value}`, 'Content-Type': 'application/json' }; }
function log(message) { logOutput.textContent = `${new Date().toLocaleTimeString()} ${message}\\n` + logOutput.textContent; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...headers(), ...(options.headers || {}) } });
  const text = await response.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
  if (!response.ok) throw new Error(body.error?.message || text || response.statusText);
  return body;
}
function statusClass(s) { return ['healthy','unhealthy','cooldown','disabled'].includes(s) ? s : ''; }

function renderProxyPool(pool) {
  if (!pool) { proxyPoolSection.innerHTML = '<div class="muted">No proxy pool data.</div>'; return; }
  const total    = pool.total    ?? pool.count ?? 0;
  const enabled  = pool.enabled  ?? total;
  const disabled = pool.disabled ?? 0;
  const staticC  = pool.static   ?? 0;
  const rotatingC= pool.rotating ?? 0;
  const file     = pool.file     || 'proxies.txt';
  const entries  = pool.entries  || [];

  // Build rows for every proxy (static + rotating)
  const rows = entries.map(e => {
    const isEnabled  = e.enabled !== false;
    const kindTag    = `<span class="tag tag-${e.kind}">${e.kind}</span>`;
    const enabledTag = isEnabled
      ? '<span class="tag tag-enabled">enabled</span>'
      : '<span class="tag tag-disabled">disabled</span>';

    let extraCols = '';
    if (e.kind === 'rotating') {
      const readyTag = e.ready_to_rotate
        ? '<span class="tag tag-ready">ready</span>'
        : `<span class="tag tag-cooldown">${e.seconds_until_next_rotation}s</span>`;
      extraCols = `<td>${e.min_interval_seconds}s</td><td>${readyTag}</td>
        <td><button class="sm warning" onclick="rotateProxy('${e.id}')">Rotate IP</button></td>`;
    } else {
      extraCols = `<td colspan="3" class="muted">—</td>`;
    }

    const toggleBtn = isEnabled
      ? `<button class="sm danger" onclick="setProxyEnabled('${e.id}', false)">Disable</button>`
      : `<button class="sm good"  onclick="setProxyEnabled('${e.id}', true)">Enable</button>`;

    return `<tr class="${isEnabled ? '' : 'disabled-row'}">
      <td style="font-family:monospace;font-size:12px">${escapeHtml(e.proxy)}</td>
      <td>${kindTag}</td>
      <td>${enabledTag}</td>
      ${extraCols}
      <td>${toggleBtn}</td>
    </tr>`;
  }).join('');

  const hasRotating = rotatingC > 0;

  proxyPoolSection.innerHTML = `
    <div class="card">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px;">
        <span style="font-size:22px;font-weight:700;">${total}</span>
        <span class="muted">proxies from <code>${escapeHtml(file)}</code></span>
        <span class="tag tag-static">${staticC} static</span>
        <span class="tag tag-rotating">${rotatingC} rotating</span>
        <span class="tag tag-enabled">${enabled} enabled</span>
        ${disabled > 0 ? `<span class="tag tag-disabled">${disabled} disabled</span>` : ''}
        ${hasRotating ? `<button class="warning sm" style="margin-left:auto" onclick="rotateAll()">Rotate all ready IPs</button>` : ''}
      </div>
      ${entries.length === 0
        ? '<div class="muted">No proxies loaded.</div>'
        : `<table>
            <tr>
              <th>Proxy</th><th>Kind</th><th>State</th>
              <th>Min interval</th><th>Rotation</th><th>Actions</th><th></th>
            </tr>
            ${rows}
          </table>`}
    </div>`;
}

async function loadAll() {
  try {
    const status = await api('/v0/management/status');
    strategyInput.value = status.rotation_strategy;
    const pool = status.proxy_pool || {};
    const total    = pool.total    ?? pool.count ?? 0;
    const staticC  = pool.static   ?? 0;
    const rotatingC= pool.rotating ?? 0;
    const enabledC = pool.enabled  ?? total;
    summaryOutput.innerHTML = `
      <div class="card"><strong>${status.token_count}</strong><div class="muted">managed tokens</div></div>
      <div class="card"><strong>${status.rotation_strategy}</strong><div class="muted">rotation strategy</div></div>
      <div class="card"><strong>${status.supervisor_running}</strong><div class="muted">health supervisor running</div></div>
      <div class="card">
        <strong>${staticC} static · ${rotatingC} rotating</strong>
        <div class="muted">${total} proxies · ${enabledC} enabled</div>
      </div>
      <div class="card"><strong>${JSON.stringify(status.counts)}</strong><div class="muted">status counts</div></div>`;

    // Fetch full proxy pool (includes all entries with ids)
    let poolData = pool;
    if (!pool.entries) {
      try { poolData = await api('/v0/management/proxies'); } catch {}
    }
    if (poolData) poolData.file = pool.file || poolData.file;
    renderProxyPool(poolData);

    const tokens = await api('/v0/management/tokens');
    tokensOutput.innerHTML = tokens.data.map(token => `
      <div class="card">
        <h3>${escapeHtml(token.name)}</h3>
        <div><span class="status ${statusClass(token.status)}">${token.status}</span></div>
        <p class="muted">${escapeHtml(token.redacted_token)} · ${escapeHtml(token.fingerprint)}</p>
        <p class="muted">proxy ${escapeHtml(token.proxy || 'Direct')}</p>
        <p class="muted">success ${token.success_count} · failure ${token.failure_count} · in flight ${token.in_flight}</p>
        ${token.last_error ? `<p class="muted">last error: ${escapeHtml(token.last_error)}</p>` : ''}
        <button onclick="checkToken('${token.id}')">Check</button>
        <button class="secondary" onclick="renameToken('${token.id}', '${escapeAttr(token.name)}')">Rename</button>
        <button class="secondary" onclick="replaceToken('${token.id}')">Replace token</button>
        <button class="secondary" onclick="toggleToken('${token.id}', ${!token.enabled})">${token.enabled ? 'Disable' : 'Enable'}</button>
        <button class="warning" onclick="rotateTokenProxy('${token.id}')">Rotate IP</button>
        <button class="danger" onclick="deleteToken('${token.id}')">Delete</button>
      </div>`).join('') || '<div class="muted">No managed tokens yet.</div>';
    log('Refreshed dashboard.');
  } catch (error) { log(`Error: ${error.message}`); }
}
async function setProxyEnabled(proxyId, enabled) {
  try {
    const result = await api(`/v0/management/proxies/${proxyId}`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    });
    log(`Proxy ${result.proxy || proxyId}: ${enabled ? 'enabled ✓' : 'disabled ✗'}`);
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function addToken(event) {
  event.preventDefault();
  try {
    const payload = { name: nameInput.value, token: tokenInput.value, check: checkInput.checked };
    if (proxyInput.value) payload.proxy = proxyInput.value;
    await api('/v0/management/tokens', { method: 'POST', body: JSON.stringify(payload) });
    nameInput.value = ''; tokenInput.value = ''; proxyInput.value = ''; checkInput.checked = false;
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function setRotation(event) {
  event.preventDefault();
  try { await api('/v0/management/rotation', { method: 'PATCH', body: JSON.stringify({ strategy: strategyInput.value }) }); await loadAll(); }
  catch (error) { log(`Error: ${error.message}`); }
}
async function rotateAll() {
  try {
    const result = await api('/v0/management/proxies/rotate', { method: 'POST' });
    const entries = Object.entries(result.results || {});
    if (entries.length === 0) { log('Rotate: no rotating proxies were ready (still in cooldown).'); }
    else { entries.forEach(([proxy, ok]) => log(`Rotate ${proxy}: ${ok ? 'success ✓' : 'failed ✗'}`)); }
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function rotateProxy(proxyId) {
  try {
    // Find proxy URL from current pool entries to call rotate_for_proxy_url via rotate-all workaround
    // Use /proxies/rotate which only rotates ready ones; we trigger individually via token endpoint if possible
    const result = await api('/v0/management/proxies/rotate', { method: 'POST' });
    const entries = Object.entries(result.results || {});
    if (entries.length === 0) { log('Rotate: proxy not ready yet (cooldown).'); }
    else { entries.forEach(([proxy, ok]) => log(`Rotate ${proxy}: ${ok ? 'success ✓' : 'failed ✗'}`)); }
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function rotateTokenProxy(id) {
  try {
    const result = await api(`/v0/management/tokens/${id}/rotate-proxy`, { method: 'POST' });
    log(`Rotate token proxy: ${result.message} (${result.proxy})`);
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function checkToken(id) { try { await api(`/v0/management/tokens/${id}/check`, { method: 'POST' }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } }
async function checkAll() { try { await api('/v0/management/tokens/check', { method: 'POST' }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } }
async function toggleToken(id, enabled) { try { await api(`/v0/management/tokens/${id}`, { method: 'PATCH', body: JSON.stringify({ enabled }) }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } }
async function renameToken(id, current) { const newName = prompt('Token name', current); if (newName !== null) { try { await api(`/v0/management/tokens/${id}`, { method: 'PATCH', body: JSON.stringify({ name: newName }) }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } } }
async function replaceToken(id) { const replacement = prompt('Paste the replacement DeepSeek web token'); if (replacement) { try { await api(`/v0/management/tokens/${id}`, { method: 'PATCH', body: JSON.stringify({ token: replacement }) }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } } }
async function deleteToken(id) { if (confirm('Delete this token?')) { try { await api(`/v0/management/tokens/${id}`, { method: 'DELETE' }); await loadAll(); } catch (error) { log(`Error: ${error.message}`); } } }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function escapeAttr(value) { return escapeHtml(value).replace(/`/g, '&#96;'); }
loadAll();
</script>
</body>
</html>
"""
