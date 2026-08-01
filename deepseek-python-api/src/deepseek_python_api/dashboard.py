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
    button { border: 0; border-radius: 10px; background: #2563eb; color: white; padding: 10px 12px; margin: 6px 6px 0 0; cursor: pointer; }
    button.secondary { background: #475569; }
    button.danger { background: #dc2626; }
    button.good { background: #16a34a; }
    .muted { color: #94a3b8; font-size: 13px; }
    .status { display: inline-block; border-radius: 999px; padding: 3px 9px; font-size: 12px; background: #475569; }
    .healthy { background: #166534; }
    .unhealthy, .disabled { background: #991b1b; }
    .cooldown { background: #92400e; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #020617; padding: 12px; border-radius: 12px; color: #cbd5e1; }
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
const logOutput = document.getElementById('log');
const summaryOutput = document.getElementById('summary');
const tokensOutput = document.getElementById('tokens');
keyInput.value = localStorage.getItem('deepseekManagementKey') || '';
function saveKey() { localStorage.setItem('deepseekManagementKey', keyInput.value); log('Saved key locally.'); }
function headers() { return { 'Authorization': `Bearer ${keyInput.value}`, 'Content-Type': 'application/json' }; }
function log(message) { logOutput.textContent = `${new Date().toLocaleTimeString()} ${message}\n` + logOutput.textContent; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...headers(), ...(options.headers || {}) } });
  const text = await response.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { raw: text }; }
  if (!response.ok) throw new Error(body.error?.message || text || response.statusText);
  return body;
}
function statusClass(status) { return ['healthy','unhealthy','cooldown','disabled'].includes(status) ? status : ''; }
async function loadAll() {
  try {
    const status = await api('/v0/management/status');
    strategyInput.value = status.rotation_strategy;
    summaryOutput.innerHTML = `
      <div class="card"><strong>${status.token_count}</strong><div class="muted">managed tokens</div></div>
      <div class="card"><strong>${status.rotation_strategy}</strong><div class="muted">rotation strategy</div></div>
      <div class="card"><strong>${status.supervisor_running}</strong><div class="muted">health supervisor running</div></div>
      <div class="card"><strong>${JSON.stringify(status.counts)}</strong><div class="muted">status counts</div></div>`;
    const tokens = await api('/v0/management/tokens');
    tokensOutput.innerHTML = tokens.data.map(token => `
      <div class="card">
        <h3>${escapeHtml(token.name)}</h3>
        <div><span class="status ${statusClass(token.status)}">${token.status}</span></div>
        <p class="muted">${escapeHtml(token.redacted_token)} · ${escapeHtml(token.fingerprint)}</p>
        <p class="muted">success ${token.success_count} · failure ${token.failure_count} · in flight ${token.in_flight}</p>
        ${token.last_error ? `<p class="muted">last error: ${escapeHtml(token.last_error)}</p>` : ''}
        <button onclick="checkToken('${token.id}')">Check</button>
        <button class="secondary" onclick="renameToken('${token.id}', '${escapeAttr(token.name)}')">Rename</button>
        <button class="secondary" onclick="replaceToken('${token.id}')">Replace token</button>
        <button class="secondary" onclick="toggleToken('${token.id}', ${!token.enabled})">${token.enabled ? 'Disable' : 'Enable'}</button>
        <button class="danger" onclick="deleteToken('${token.id}')">Delete</button>
      </div>`).join('') || '<div class="muted">No managed tokens yet.</div>';
    log('Refreshed dashboard.');
  } catch (error) { log(`Error: ${error.message}`); }
}
async function addToken(event) {
  event.preventDefault();
  try {
    await api('/v0/management/tokens', { method: 'POST', body: JSON.stringify({ name: nameInput.value, token: tokenInput.value, check: checkInput.checked }) });
    nameInput.value = ''; tokenInput.value = ''; checkInput.checked = false;
    await loadAll();
  } catch (error) { log(`Error: ${error.message}`); }
}
async function setRotation(event) {
  event.preventDefault();
  try { await api('/v0/management/rotation', { method: 'PATCH', body: JSON.stringify({ strategy: strategyInput.value }) }); await loadAll(); }
  catch (error) { log(`Error: ${error.message}`); }
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
