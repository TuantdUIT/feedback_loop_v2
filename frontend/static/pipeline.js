// Tab "Mô phỏng Production": upload → theo dõi run → metrics → từng case.
(() => {
  const $ = sel => document.querySelector(sel);
  const DECISIONS = {
    agree: 'Đồng thuận', model_better: 'Model tốt hơn', deepseek_better: 'DeepSeek tốt hơn', tie: 'Hoà',
    inconclusive: 'Không kết luận (lật)', deepseek_invalid: 'DeepSeek sai hình thức', failed: 'Lỗi API',
    budget_exceeded: 'Chạm trần chi phí',
  };
  const FLOW = [
    { key: 'ingested', name: 'Chờ DeepSeek', topic: 'ingested' },
    { key: 'parsed', name: 'Chờ so sánh', topic: 'parsed' },
    { key: 'to_judge', name: 'Chờ judge', topic: 'to_judge' },
    { key: 'gate', name: 'Chờ gate', stages: ['agreed', 'judged'] },
    { key: 'decided', name: 'Xong', topic: 'decided' },
  ];
  const state = { upload: null, gold: null, runId: null, run: null, cases: [], filter: 'all', caseId: null, timer: null };

  function el(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined && content !== null) node.textContent = String(content);
    return node;
  }
  const usd = v => (v === null || v === undefined) ? '—' : `$${Number(v).toFixed(v < 0.01 ? 5 : 4)}`;
  const pct = v => (v === null || v === undefined) ? '—' : `${(v * 100).toFixed(1)}%`;
  const ci = r => (r && r.ci95) ? ` [${pct(r.ci95[0])}–${pct(r.ci95[1])}]` : '';
  const when = t => t ? new Date(t * 1000).toLocaleString('vi-VN') : '—';

  async function api(path, options = {}) {
    const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }

  // ------------------------------------------------------------------ upload
  const fileInput = $('#pl-file');
  const goldInput = $('#pl-gold');
  const calibration = $('#pl-calibration');
  const runButton = $('#pl-run-button');
  const preview = $('#pl-preview');

  calibration.addEventListener('change', () => {
    $('#pl-gold-wrap').hidden = !calibration.checked;
    refreshRunButton();
  });

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    state.upload = null;
    refreshRunButton();
    if (!file) return;
    const content = await file.text();
    preview.hidden = false;
    preview.replaceChildren(el('span', 'pl-muted', 'Đang đọc file…'));
    try {
      const result = await api('/api/runs/preview', { method: 'POST', body: JSON.stringify({ filename: file.name, content }) });
      state.upload = { filename: file.name, content, preview: result };
      renderPreview(result);
    } catch (error) {
      preview.replaceChildren(el('span', 'pl-error', `Không đọc được file: ${error.message}`));
    }
    refreshRunButton();
  });

  goldInput.addEventListener('change', async () => {
    state.gold = null;
    const file = goldInput.files[0];
    if (file) {
      try { state.gold = JSON.parse(await file.text()); } catch { alert('File gold không phải JSON hợp lệ'); }
    }
    refreshRunButton();
  });

  function kv(value, label) {
    const node = el('div', 'pl-kv');
    node.append(el('strong', null, value), el('span', null, label));
    return node;
  }

  function renderPreview(p) {
    const items = [
      kv(p.n_cases, 'case hợp lệ'),
      kv(p.skipped.length, 'message bị loại'),
      kv(p.model_layer0_hard, 'case model sai hình thức'),
      kv(usd(p.estimate.deepseek_usd), 'DeepSeek ước tính'),
      kv(`≤ ${usd(p.estimate.total_max_usd)}`, 'trần ước tính (mọi case qua judge)'),
      kv(usd(p.budget_usd), 'trần chi phí / run'),
    ];
    preview.replaceChildren(...items);
    if (p.estimate.total_max_usd > p.budget_usd) {
      preview.append(el('div', 'pl-error', 'Trần ước tính vượt ngân sách: pipeline sẽ dừng gọi API khi chạm trần, các case còn lại nhận "Chạm trần chi phí".'));
    }
    $('#pl-run-cost').textContent = `≤ ${usd(Math.min(p.estimate.total_max_usd, p.budget_usd))}`;
  }

  function refreshRunButton() {
    const ready = state.upload && state.upload.preview.n_cases > 0 && (!calibration.checked || state.gold);
    runButton.disabled = !ready;
    $('#pl-upload-hint').textContent = calibration.checked && !state.gold
      ? 'Lượt hiệu chuẩn cần file gold.'
      : ready ? 'Bấm Chạy để đưa các case vào queue. Mỗi case gọi DeepSeek; case khác nhau gọi thêm Qwen 2 lần.'
        : 'Chọn file để xem số case và chi phí ước tính trước khi chạy.';
  }

  runButton.addEventListener('click', async () => {
    runButton.disabled = true;
    try {
      const body = { filename: state.upload.filename, content: state.upload.content,
        kind: calibration.checked ? 'calibration' : 'normal', gold: calibration.checked ? state.gold : null };
      const { run_id } = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) });
      fileInput.value = ''; goldInput.value = ''; state.upload = null; preview.hidden = true;
      $('#pl-run-cost').textContent = '—';
      await loadRuns();
      openRun(run_id);
    } catch (error) {
      alert(`Không chạy được: ${error.message}`);
    }
    refreshRunButton();
  });

  // -------------------------------------------------------------------- runs
  async function loadConfig() {
    try {
      const cfg = await api('/api/pipeline/config');
      $('#pl-config').textContent = cfg.error ? `Cấu hình lỗi: ${cfg.error}`
        : `Judge ${cfg.judge_config.model} · vân tay ${cfg.judge_fingerprint} · ${cfg.calibrated ? 'đã hiệu chuẩn' : 'chưa hiệu chuẩn'}`;
    } catch (error) {
      $('#pl-config').textContent = `Pipeline không phản hồi: ${error.message}`;
    }
  }

  async function loadRuns() {
    let runs;
    try { runs = await api('/api/runs'); } catch (error) {
      $('#pl-runs').replaceChildren(el('p', 'pl-muted', `Không tải được: ${error.message}`));
      return;
    }
    if (!runs.length) {
      $('#pl-runs').replaceChildren(el('p', 'pl-muted', 'Chưa có lần upload nào.'));
      return;
    }
    const table = el('table', 'pl-table');
    const head = el('tr');
    ['Run', 'File', 'Loại', 'Trạng thái', 'Case', 'Chi phí / trần', 'Bắt đầu'].forEach(h => head.append(el('th', null, h)));
    table.append(head);
    for (const run of runs) {
      const row = el('tr', `pl-click${run.run_id === state.runId ? ' pl-selected' : ''}`);
      row.append(el('td', null, run.run_id), el('td', null, run.filename));
      const kind = el('td'); kind.append(el('span', `pl-badge ${run.kind}`, run.kind === 'calibration' ? 'hiệu chuẩn' : 'thường'));
      const status = el('td'); status.append(el('span', `pl-badge ${run.status}`, run.status === 'done' ? 'xong' : 'đang chạy'));
      row.append(kind, status, el('td', 'pl-num', run.n_cases),
        el('td', 'pl-num', `${usd(run.cost_usd)} / ${usd(run.budget_usd)}`), el('td', 'pl-muted', when(run.created_at)));
      row.addEventListener('click', () => openRun(run.run_id));
      table.append(row);
    }
    $('#pl-runs').replaceChildren(table);
  }

  function openRun(runId) {
    state.runId = runId;
    state.filter = 'all';
    closeCase();
    $('#pl-run').hidden = false;
    refreshRun();
    loadRuns();
  }

  async function refreshRun() {
    clearTimeout(state.timer);
    if (!state.runId) return;
    try {
      const [run, cases] = await Promise.all([api(`/api/runs/${state.runId}`), api(`/api/runs/${state.runId}/cases`)]);
      state.run = run; state.cases = cases;
      renderRun();
      if (run.status !== 'done') state.timer = setTimeout(refreshRun, 2000);
      else loadRuns();
    } catch (error) {
      $('#pl-run-title').textContent = `Lỗi tải run: ${error.message}`;
      state.timer = setTimeout(refreshRun, 5000);
    }
  }

  function renderRun() {
    const run = state.run;
    $('#pl-run-title').textContent = run.run_id;
    const status = $('#pl-run-status');
    status.className = `pl-badge ${run.status}`;
    status.textContent = run.status === 'done' ? 'xong' : 'đang chạy';
    $('#pl-run-meta').textContent = `${run.filename} · ${run.kind === 'calibration' ? 'hiệu chuẩn' : 'thường'} · prompt ${run.meta.prompt_sha} · judge ${run.meta.judge_fingerprint}`;
    const queue = $('#pl-label-queue');
    queue.href = `/api/runs/${run.run_id}/label-queue`;
    queue.download = `${run.run_id}_label_queue.json`;
    renderFlow(run);
    renderBudget(run);
    renderRecommendation(run.report);
    renderMetrics(run.report);
    renderFilters();
    renderCases();
    if (state.caseId) openCase(state.caseId, false);
  }

  function renderFlow(run) {
    const stages = run.progress.stages;
    const processing = run.progress.processing;
    const nodes = FLOW.map(step => {
      const count = step.stages ? step.stages.reduce((s, k) => s + (stages[k] || 0), 0) : (stages[step.key] || 0);
      const busy = step.stages ? step.stages.reduce((s, k) => s + (processing[k] || 0), 0) : (processing[step.topic] || 0);
      const node = el('div', `pl-step${busy ? ' active' : ''}`);
      node.append(el('div', 'pl-step-name', step.name), el('div', 'pl-step-count', count),
        el('div', 'pl-step-sub', busy ? `${busy} đang xử lý` : ''));
      const bar = el('div', 'pl-step-bar'); bar.style.width = `${run.n_cases ? (count / run.n_cases) * 100 : 0}%`;
      node.append(bar);
      return node;
    });
    const failed = Object.values(run.progress.failed_messages || {}).reduce((a, b) => a + b, 0);
    const last = el('div', 'pl-step');
    last.append(el('div', 'pl-step-name', 'Message lỗi'), el('div', 'pl-step-count', failed), el('div', 'pl-step-sub', failed ? 'xem cột Quyết định' : ''));
    $('#pl-flow').replaceChildren(...nodes, last);
  }

  function renderBudget(run) {
    const ratio = run.budget_usd ? run.cost_usd / run.budget_usd : 0;
    const track = el('div', 'pl-budget-track');
    const fill = el('div', `pl-budget-fill${ratio > 0.8 ? ' high' : ''}`); fill.style.width = `${Math.min(100, ratio * 100)}%`;
    track.append(fill);
    $('#pl-budget').replaceChildren(el('span', null, 'Chi phí'), track,
      el('span', null, `${usd(run.cost_usd)} / ${usd(run.budget_usd)}`));
  }

  function renderRecommendation(report) {
    const box = $('#pl-recommendation');
    if (!report) { box.className = 'pl-recommendation'; box.replaceChildren(); return; }
    const rec = report.recommendation;
    box.className = `pl-recommendation ${rec.status}`;
    const list = el('ul');
    for (const reason of rec.reasons || []) list.append(el('li', null, reason));
    box.replaceChildren(el('strong', null, rec.label), list);
  }

  function metric(label, value, note) {
    const row = el('div', 'pl-metric');
    const left = el('span', null, label);
    if (note) left.append(el('small', null, note));
    row.append(left, el('b', null, value));
    return row;
  }

  function card(title, rows, hint) {
    const node = el('div', 'pl-card');
    node.append(el('h3', null, title), ...rows);
    if (hint) node.append(el('div', 'pl-hint', hint));
    return node;
  }

  function renderMetrics(report) {
    const box = $('#pl-metrics');
    if (!report) {
      box.replaceChildren(el('p', 'pl-muted', 'Metrics hiện khi mọi case đã xong.'));
      return;
    }
    const m = report.metrics;
    const judge = m.judge_health;
    const quality = m.model_quality;
    const cards = [
      card('Đồng thuận model ↔ DeepSeek', [
        metric('Giống hệt', pct(m.agreement.identical.rate), `${m.agreement.identical.k}/${m.agreement.identical.n} case`),
        metric('Span F1', pct(m.agreement.span_model_vs_deepseek.f1), 'DeepSeek chỉ là mốc so, không phải đáp án'),
      ], 'Giống nhau không có nghĩa là đúng — random audit lấy mẫu cả nhánh này.'),
      card('Sức khoẻ judge', [
        metric('Tỉ lệ lật', pct(judge.flip_rate.rate), `position bias · ${judge.flip_rate.k}/${judge.flip_rate.n}${ci(judge.flip_rate)}`),
        metric('Thắng khi ở vị trí 1', pct(judge.first_slot_win_rate.rate), `lý tưởng ≈ 50% · n=${judge.first_slot_win_rate.n}`),
        metric('Bên thắng nhiều span hơn', pct(judge.winner_has_more_spans_rate.rate), `verbosity · n=${judge.winner_has_more_spans_rate.n}`),
        metric('Hoà', pct(judge.tie_rate.rate), `${judge.tie_rate.k}/${judge.tie_rate.n}`),
      ], `${judge.n_judged} case qua judge.`),
      card('Chất lượng model (ước lượng)', [
        metric('DeepSeek tốt hơn', pct(quality.deepseek_better.rate), `${quality.deepseek_better.k}/${quality.deepseek_better.n}${ci(quality.deepseek_better)}`),
        metric('Lỗi đã xác nhận', quality.confirmed_error.n ? pct(quality.confirmed_error.rate) : '—', 'cần ngưỡng severity đã hiệu chuẩn'),
        metric('Kiểu lỗi', Object.entries(quality.error_kinds).map(([k, v]) => `${k} ${v}`).join(' · ') || '—', 'INC nhãn · PAR biên · MIS thiếu · SPU thừa'),
        metric('Level liên quan', Object.entries(quality.levels_involved).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'),
      ]),
      card('Chi phí & độ trễ', [
        metric('DeepSeek', usd(m.cost.deepseek_usd), m.latency_sec.deepseek ? `TB ${m.latency_sec.deepseek.mean}s · p95 ${m.latency_sec.deepseek.p95}s` : ''),
        metric('Qwen judge', usd(m.cost.judge_usd), m.latency_sec.judge ? `TB ${m.latency_sec.judge.mean}s · p95 ${m.latency_sec.judge.p95}s` : ''),
        metric('Tổng', usd(m.cost.cases_total_usd)),
        metric('Case audit', m.audit.n_flagged, 'đưa vào hàng chờ gán nhãn tay'),
      ]),
    ];
    if (report.calibration_run) {
      const c = report.calibration_run;
      cards.push(card('Hiệu chuẩn (so với gold)', [
        metric('Judge chọn đúng', pct(c.accuracy.rate), `${c.accuracy.k}/${c.accuracy.n}`),
        metric('Cohen κ', c.kappa === null ? '—' : c.kappa.toFixed(3)),
        metric('Báo động giả', pct(c.false_alarm.rate), `${c.false_alarm.k}/${c.false_alarm.n}`),
        metric('Bắt lỗi · precision / recall', `${pct(c.detect_precision.rate)} / ${pct(c.detect_recall.rate)}`),
        metric('Đồng thuận nhưng cùng sai', pct(c.agree_but_wrong.rate), `${c.agree_but_wrong.k}/${c.agree_but_wrong.n}`),
      ]));
    }
    box.replaceChildren(...cards);
  }

  // ------------------------------------------------------------------- cases
  function renderFilters() {
    const counts = {};
    for (const c of state.cases) { const k = c.decision || 'open'; counts[k] = (counts[k] || 0) + 1; }
    const chips = [['all', `Tất cả ${state.cases.length}`]];
    for (const [k, n] of Object.entries(counts)) chips.push([k, `${DECISIONS[k] || 'Đang chạy'} ${n}`]);
    const audit = state.cases.filter(c => c.audit).length;
    if (audit) chips.push(['audit', `Audit ${audit}`]);
    $('#pl-filters').replaceChildren(...chips.map(([key, label]) => {
      const chip = el('button', `pl-chip${state.filter === key ? ' active' : ''}`, label);
      chip.type = 'button';
      chip.addEventListener('click', () => { state.filter = key; renderFilters(); renderCases(); });
      return chip;
    }));
  }

  function spanChips(spans, other) {
    const box = el('div', 'pl-spans');
    if (!spans) { box.append(el('span', 'pl-muted', '—')); return box; }
    if (!spans.length) { box.append(el('span', 'pl-muted', '(rỗng)')); return box; }
    const otherKeys = new Set((other || []).map(s => `${s.level}|${s.start}|${s.end}`));
    for (const s of spans) {
      const differs = other && !otherKeys.has(`${s.level}|${s.start}|${s.end}`);
      const chip = el('span', `pl-span ${s.level}${differs ? ' diff' : ''}`);
      chip.append(el('b', null, s.level), document.createTextNode(s.text));
      box.append(chip);
    }
    return box;
  }

  function renderCases() {
    const rows = state.cases.filter(c => state.filter === 'all' || (state.filter === 'audit' ? c.audit : (c.decision || 'open') === state.filter));
    const table = el('table', 'pl-table');
    const head = el('tr');
    ['#', 'Input', 'Model đang triển khai', 'DeepSeek', 'Judge (2 lần)', 'Quyết định', 'Chi phí'].forEach(h => head.append(el('th', null, h)));
    table.append(head);
    for (const c of rows) {
      const row = el('tr', `pl-click${c.case_id === state.caseId ? ' pl-selected' : ''}`);
      const input = el('td', null, c.text);
      if (c.audit) input.append(el('span', 'pl-audit', 'audit'));
      const models = el('td'); models.append(spanChips(c.model, c.deepseek));
      const ds = el('td'); ds.append(spanChips(c.deepseek, c.model));
      const judge = el('td', 'pl-muted', c.winners ? c.winners.map(w => w === 'tie' ? 'hoà' : w).join(' · ') : (c.decision === 'agree' ? 'không cần' : '—'));
      const decision = el('td');
      decision.append(el('span', `pl-decision ${c.decision || ''}`, c.decision ? DECISIONS[c.decision] : `… ${c.stage}`));
      if (c.error) decision.append(el('div', 'pl-muted', c.error.slice(0, 80)));
      row.append(el('td', 'pl-num', c.position), input, models, ds, judge, decision, el('td', 'pl-num', usd(c.cost_usd)));
      row.addEventListener('click', () => openCase(c.case_id, true));
      table.append(row);
    }
    $('#pl-cases').replaceChildren(rows.length ? table : el('p', 'pl-muted', 'Không có case nào.'));
  }

  function highlight(text, spans) {
    const box = el('div', 'pl-hl');
    let pos = 0;
    for (const s of [...(spans || [])].sort((a, b) => a.start - b.start)) {
      if (s.start > pos) box.append(document.createTextNode(text.slice(pos, s.start)));
      const mark = el('mark', s.level, text.slice(s.start, s.end));
      mark.append(el('sup', null, s.level));
      box.append(mark);
      pos = s.end;
    }
    if (pos < text.length) box.append(document.createTextNode(text.slice(pos)));
    return box;
  }

  async function openCase(caseId, scroll) {
    state.caseId = caseId;
    let record;
    try { record = await api(`/api/runs/${state.runId}/cases/${encodeURIComponent(caseId)}`); } catch (error) {
      $('#pl-case-body').replaceChildren(el('p', 'pl-error', error.message));
      return;
    }
    $('#pl-case').hidden = false;
    $('#pl-case-title').textContent = `${caseId} · ${record.decision ? DECISIONS[record.decision] : 'đang chạy'}`;
    const body = [];
    body.push(el('h4', null, 'Model đang triển khai'), highlight(record.raw_text, record.old.spans));
    body.push(el('h4', null, 'DeepSeek'), record.new ? highlight(record.raw_text, record.new.spans) : el('p', 'pl-muted', 'Chưa có.'));

    if (record.diff && record.diff.items) {
      body.push(el('h4', null, `Khác biệt · severity ${record.diff.severity_total}`));
      const diffs = record.diff.items.filter(i => i.kind !== 'COR');
      if (!diffs.length) body.push(el('p', 'pl-muted', 'Hai bản giống hệt.'));
      for (const item of diffs) {
        const line = el('div', 'pl-issue');
        line.append(el('span', 'pl-kind', item.kind),
          document.createTextNode(`model: ${item.old ? `${item.old.level} "${item.old.text}"` : '—'}  →  DeepSeek: ${item.new ? `${item.new.level} "${item.new.text}"` : '—'}`));
        body.push(line);
      }
    }

    const judge = record.layers.judge;
    if (judge) {
      body.push(el('h4', null, `Judge · ${judge.consistent ? 'nhất quán' : 'LẬT giữa hai thứ tự'}`));
      for (const call of judge.calls) {
        const box = el('div', 'pl-call');
        const head = el('div', 'pl-call-head');
        head.append(el('span', null, `Thứ tự: 1 = ${call.order[0]}, 2 = ${call.order[1]} → chọn ${call.winner}`),
          el('span', 'pl-muted', `${usd(call.cost_usd)} · ${call.elapsed_sec}s · ${call.attempts.length} lần thử`));
        box.append(head);
        for (const side of ['model', 'deepseek']) {
          for (const issue of call.issues[side] || []) {
            box.append(el('div', 'pl-issue', `${side}: ${issue.problem} · ${issue.level || ''} "${issue.text}" · ${issue.rule || ''}`));
          }
        }
        body.push(box);
      }
    }
    const gate = record.layers.gate;
    if (gate) {
      body.push(el('h4', null, 'Gate'));
      body.push(el('p', null, gate.confirmed_error === null ? 'Ứng viên lỗi — chưa có ngưỡng severity đã hiệu chuẩn.'
        : gate.confirmed_error ? 'Lỗi đã xác nhận của model.' : 'Không phải lỗi đã xác nhận.'));
    }
    const l0 = record.layers.layer0_model;
    const l0ds = record.layers.deepseek && record.layers.deepseek.layer0;
    body.push(el('h4', null, 'Lớp 0 (hình thức)'));
    body.push(el('p', null, `Model: ${l0 && l0.ok ? 'hợp lệ' : `${(l0 && l0.hard.length) || 0} vi phạm hard`} · DeepSeek: ${l0ds ? (l0ds.ok ? 'hợp lệ' : `${l0ds.hard.length} vi phạm hard`) : '—'}`));
    body.push(el('h4', null, 'Chi phí'), el('p', null, usd(record.cost_usd)));
    const raw = el('details');
    raw.append(el('summary', 'pl-muted', 'CaseRecord đầy đủ (JSON)'), el('pre', 'pl-json', JSON.stringify(record, null, 2)));
    body.push(raw);
    $('#pl-case-body').replaceChildren(...body);
    if (scroll) renderCases();
  }

  function closeCase() {
    state.caseId = null;
    $('#pl-case').hidden = true;
  }
  $('#pl-case-close').addEventListener('click', () => { closeCase(); renderCases(); });

  document.getElementById('tab-btn-prod').addEventListener('click', () => { loadConfig(); loadRuns(); });
  loadConfig();
  loadRuns();
})();
