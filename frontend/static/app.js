const ENGINES = [
  { key: 'pytorch_fp32', route: 'pytorch', name: 'PyTorch FP32', kind: 'Local · PhoBERT' },
  { key: 'onnx_fp32', route: 'onnx', name: 'ONNX FP32', kind: 'Local · PhoBERT' },
  { key: 'qwen', route: 'qwen', name: 'Qwen', kind: 'OpenRouter · có phí' },
];

const form = document.querySelector('#parse-form');
const input = document.querySelector('#address-input');
const button = document.querySelector('#parse-button');
const chainList = document.querySelector('#chain-list');
const engineGrid = document.querySelector('#engine-grid');
const staleBanner = document.querySelector('#stale-banner');
const results = Object.fromEntries(ENGINES.map(engine => [engine.key, { phase: 'idle' }]));
let submittedText = null;
let pending = false;
let clock = null;

function el(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined && content !== null) node.textContent = String(content);
  return node;
}

function stateText(record) {
  if (record.phase === 'idle') return '(chưa chạy)';
  if (record.phase === 'waiting') return `⏳ Đang chờ… ${Math.floor((Date.now() - record.started) / 1000)}s`;
  if (record.phase === 'error') return '⚠ lỗi';
  if (!record.data.spans.length) return '(rỗng)';
  return null;
}

function stateClass(record) {
  return record.phase === 'waiting' ? 'waiting' : record.phase === 'error' ? 'error' : record.phase === 'done' ? 'empty' : '';
}

function timeCost(record) {
  if (!record.data) return '';
  return `${Number(record.data.elapsed_sec || 0).toFixed(3)}s · $${Number(record.data.cost_usd || 0).toFixed(5)}`;
}

function groups() {
  const chains = [...new Set(ENGINES.map(engine => results[engine.key])
    .filter(record => record.phase === 'done' && record.data.spans.length)
    .map(record => record.data.level_chain))];
  return Object.fromEntries(chains.map((chain, index) => [chain, `group-${'abc'[index % 3]}`]));
}

function renderChains() {
  chainList.replaceChildren();
  const colors = groups();
  for (const engine of ENGINES) {
    const record = results[engine.key];
    const row = el('div', 'chain-row');
    const title = el('div');
    title.append(el('div', 'engine-name', engine.name), el('div', 'engine-subtitle', engine.kind));
    const value = el('div', `chain-value ${record.data ? (colors[record.data.level_chain] || '') : ''}`);
    const state = stateText(record);
    if (state) value.append(el('span', `state-text ${stateClass(record)}`, state));
    else record.data.spans.forEach((span, index) => {
      if (index) value.append(el('span', 'chain-arrow', '›'));
      value.append(el('span', 'level', span.level ?? '—'));
    });
    row.append(title, value, el('div', 'chain-meta', timeCost(record)));
    chainList.append(row);
  }
}

function spanTable(spans) {
  const table = el('table', 'span-table');
  const head = el('thead');
  const headerRow = el('tr');
  ['Level', 'Text', 'start:end', 'truncated'].forEach(label => headerRow.append(el('th', '', label)));
  head.append(headerRow);
  const body = el('tbody');
  for (const span of spans) {
    const row = el('tr');
    row.append(
      el('td', 'span-level', span.level ?? '—'),
      el('td', 'span-text', span.text ?? '—'),
      el('td', 'mono', span.start == null || span.end == null ? '—' : `${span.start}:${span.end}`),
      el('td', 'mono', span.truncated == null ? '—' : span.truncated ? 'true' : 'false'),
    );
    body.append(row);
  }
  table.append(head, body);
  return table;
}

function renderCards() {
  engineGrid.replaceChildren();
  for (const engine of ENGINES) {
    const record = results[engine.key];
    const card = el('article', 'engine-card');
    const header = el('div', 'engine-card-header');
    const heading = el('div');
    heading.append(el('h3', '', engine.name), el('small', '', engine.kind));
    header.append(heading, el('span', `engine-badge ${engine.key === 'qwen' ? 'paid' : ''}`, engine.key === 'qwen' ? 'PAID API' : 'LOCAL'));
    card.append(header);
    if (record.phase === 'done' && record.data.spans.length) {
      card.append(spanTable(record.data.spans));
    } else {
      card.append(el('div', `card-placeholder state-text ${stateClass(record)}`, stateText(record)));
    }
    if (record.phase === 'error') card.append(el('div', 'error-detail', record.data?.error || 'Không thể kết nối đến server.'));
    const footer = el('div', 'card-foot');
    footer.append(el('span', '', record.data ? `${record.data.spans?.length ?? 0} spans` : '— spans'),
      el('span', '', timeCost(record) || '— s · — USD'));
    card.append(footer);
    engineGrid.append(card);
  }
}

function renderInspector() {
  for (const engine of ENGINES.slice(0, 2)) {
    const target = document.querySelector(`.inspector-body[data-engine="${engine.key}"]`);
    const record = results[engine.key];
    target.replaceChildren();
    if (record.phase !== 'done') {
      target.append(el('div', 'inspector-empty', stateText(record) || 'Không có dữ liệu.'));
      continue;
    }
    const cols = el('div', 'inspector-columns');
    const rawCol = el('div');
    rawCol.append(el('div', 'inspector-label', 'Nhãn thô từ model'));
    const rawList = el('div', 'raw-list');
    for (const item of record.data.raw_labels || []) {
      const line = el('div', 'raw-item');
      line.append(el('span', '', item.token), el('code', '', item.label));
      rawList.append(line);
    }
    if (!rawList.childElementCount) rawList.append(el('div', 'inspector-empty', '(rỗng)'));
    rawCol.append(rawList);
    const normCol = el('div');
    normCol.append(el('div', 'inspector-label', 'Sau adapter R1–R6'));
    const normList = el('div', 'normalized-list');
    for (const span of record.data.spans) {
      const line = el('div', 'normalized-item');
      line.append(el('strong', '', span.level), el('span', '', span.text));
      normList.append(line);
    }
    if (!normList.childElementCount) normList.append(el('div', 'inspector-empty', '(rỗng)'));
    normCol.append(normList);
    cols.append(rawCol, normCol);
    target.append(cols);
  }
}

function render() {
  renderChains();
  renderCards();
  renderInspector();
  staleBanner.hidden = submittedText === null || input.value.trim() === submittedText;
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Không lấy được trạng thái');
    const status = await response.json();
    for (const [key, id, name] of [
      ['pytorch_fp32', 'load-pytorch', 'PyTorch'], ['onnx_fp32', 'load-onnx', 'ONNX'],
    ]) {
      const node = document.getElementById(id);
      const loaded = Boolean(status.loaded[key]);
      node.textContent = `${name} · ${loaded ? 'đã nạp' : 'chưa nạp'}`;
      node.classList.toggle('loaded', loaded);
    }
    const qwen = document.querySelector('#load-qwen');
    qwen.textContent = `Qwen · ${status.openrouter_configured ? 'đã cấu hình' : 'chưa cấu hình'}`;
    qwen.classList.toggle('missing', !status.openrouter_configured);
    const ram = status.memory;
    document.querySelector('#ram-status').textContent = ram?.process_mb == null
      ? 'RAM · —' : `RAM server ${ram.process_mb} MB · còn ${ram.available_mb} MB`;
  } catch {
    document.querySelector('#load-qwen').textContent = 'Không lấy được trạng thái server';
  }
}

async function parseEngine(engine, text) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), engine.key === 'qwen' ? 930000 : 180000);
    let response;
    try {
      response = await fetch(`/api/parse/${engine.route}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }), signal: controller.signal,
      });
    } finally { clearTimeout(timeout); }
    if (!response.ok) throw new Error(`Server trả HTTP ${response.status}`);
    const data = await response.json();
    results[engine.key] = { phase: data.ok ? 'done' : 'error', data };
  } catch (error) {
    results[engine.key] = { phase: 'error', data: { spans: [], error: error.name === 'AbortError'
      ? `Hết thời gian chờ ${engine.key === 'qwen' ? 930 : 180} giây` : error.message } };
  }
  render();
  if (engine.key !== 'qwen') refreshStatus();
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (pending) return;
  const text = input.value.trim();
  if (!text) { input.focus(); return; }
  pending = true;
  submittedText = text;
  button.disabled = true;
  const started = Date.now();
  for (const engine of ENGINES) results[engine.key] = { phase: 'waiting', started };
  render();
  clock = setInterval(() => { renderChains(); renderCards(); }, 1000);
  try {
    await Promise.allSettled(ENGINES.map(engine => parseEngine(engine, text)));
  } finally {
    clearInterval(clock);
    clock = null;
    pending = false;
    button.disabled = false;
    render();
    refreshStatus();
  }
});

input.addEventListener('input', () => { staleBanner.hidden = submittedText === null || input.value.trim() === submittedText; });
render();
refreshStatus();

const TABS = [
  { btn: 'tab-btn-test', panel: 'tab-panel-test' },
  { btn: 'tab-btn-prod', panel: 'tab-panel-prod' },
];
for (const tab of TABS) {
  document.getElementById(tab.btn).addEventListener('click', () => {
    for (const other of TABS) {
      const active = other.btn === tab.btn;
      document.getElementById(other.btn).classList.toggle('active', active);
      document.getElementById(other.btn).setAttribute('aria-selected', String(active));
      document.getElementById(other.panel).hidden = !active;
    }
  });
}
