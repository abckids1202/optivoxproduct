(function () {
  const $ = (id) => document.getElementById(id);
  const sevColor = { 3: 'var(--danger)', 2: 'var(--warn)', 1: 'var(--neon)', 0: 'var(--text-dim)' };
  const toggleLabels = {
    show_object_boxes: 'Object boxes',
    show_heatmap: 'Heatmap',
    show_pose_landmarks: 'Pose overlay',
    show_age_gender: 'Age/gender',
    show_zones_grid: 'Zones grid',
    danger_detection: 'Danger detection',
  };
  let togglesReady = false;
  let people = [];

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.status);
    return res.json();
  }

  async function postJSON(url, data) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data || {}),
    });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(out.message || out.answer || out.result || res.status);
    return out;
  }

  function timeAgo(iso) {
    if (!iso) return '-';
    const clean = String(iso).endsWith('Z') ? iso : `${iso}Z`;
    const seconds = (Date.now() - new Date(clean).getTime()) / 1000;
    if (!Number.isFinite(seconds)) return iso;
    if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  }

  function line(text) {
    const div = document.createElement('div');
    div.className = 'line';
    div.innerHTML = text;
    return div;
  }

  function setMsg(text, ok) {
    const el = $('actionMsg');
    if (!el) return;
    el.style.color = ok ? 'var(--ok)' : 'var(--warn)';
    el.textContent = text || '';
  }

  function renderList(id, items, empty, formatter) {
    const el = $(id);
    if (!el) return;
    el.innerHTML = '';
    if (!items || !items.length) {
      el.appendChild(line(empty));
      return;
    }
    items.forEach((item) => el.appendChild(line(formatter(item))));
  }

  function renderToggles(toggles) {
    const grid = $('toggleGrid');
    if (!grid || !toggles || togglesReady) return;
    grid.innerHTML = '';
    Object.keys(toggleLabels).forEach((key) => {
      const label = document.createElement('label');
      label.className = 'toggle';
      label.innerHTML = `<span>${toggleLabels[key]}</span><input type="checkbox" data-toggle="${key}" ${toggles[key] ? 'checked' : ''}>`;
      grid.appendChild(label);
    });
    grid.addEventListener('change', async (event) => {
      const input = event.target.closest('input[data-toggle]');
      if (!input) return;
      try {
        const out = await postJSON('/api/toggle', { name: input.dataset.toggle, value: input.checked });
        setMsg(out.message || 'updated', true);
      } catch (err) {
        input.checked = !input.checked;
        setMsg(String(err.message || err), false);
      }
    });
    togglesReady = true;
  }

  function syncToggleValues(toggles) {
    if (!toggles) return;
    Object.entries(toggles).forEach(([key, value]) => {
      const input = document.querySelector(`input[data-toggle="${key}"]`);
      if (input && document.activeElement !== input) input.checked = !!value;
    });
  }

  async function refreshStats() {
    try {
      const stats = await getJSON('/api/stats');
      $('kFaces').textContent = stats.enrolled_faces ?? '0';
      $('kEvents').textContent = stats.events_today ?? '0';
      $('kAttend').textContent = stats.clocked_in ?? '0';
      $('kStrangers').textContent = stats.active_strangers ?? '0';
    } catch (_) {}
  }

  async function refreshLiveState() {
    try {
      const state = await getJSON('/api/live_state');
      const mode = state.bridge?.mode || 'unknown';
      $('bridgeMode').textContent = mode.toUpperCase();
      $('bridgeMode').style.color = mode === 'live' ? 'var(--ok)' : (mode === 'synthetic' ? 'var(--warn)' : 'var(--text-dim)');
      const frame = state.frame || {};
      $('kProcess').textContent = frame.process_ms ?? state.stream_meta?.process_ms ?? '-';
      renderToggles(state.toggles);
      syncToggleValues(state.toggles);
      renderList('faceFeed', frame.faces || [], 'no faces visible', (f) => {
        const name = f.name || f.label || 'UNKNOWN';
        const conf = Number(f.confidence || 0).toFixed(2);
        const live = f.is_real === false ? ' · SPOOF' : '';
        return `<span class="e">${name}</span> · ${conf}${live}`;
      });
      renderList('objectFeed', frame.objects || [], 'no objects detected', (o) => {
        const name = o.class_name || 'object';
        const conf = Number(o.confidence || 0).toFixed(2);
        return `<span class="e">${name}</span> · ${conf} · ${o.category || 'general'}`;
      });
      renderList('heldFeed', frame.held_objects || [], 'no held objects confirmed', (o) => {
        const name = o.class_name || 'object';
        const conf = Number(o.confidence || 0).toFixed(2);
        return `<span class="e">${name}</span> · ${conf}`;
      });
    } catch (err) {
      $('bridgeMode').textContent = 'OFFLINE';
      $('bridgeMode').style.color = 'var(--danger)';
    }
  }

  async function refreshEvents() {
    try {
      const out = await getJSON('/api/events?limit=30');
      renderList('eventFeed', out.events || [], 'no events recorded yet', (e) => {
        const col = sevColor[e.severity] || 'var(--text-dim)';
        return `<span class="t">${timeAgo(e.timestamp)}</span> · <span class="e" style="color:${col}">${e.event_type}</span>${e.person ? ` · ${e.person}` : ''}${e.location ? ` · ${e.location}` : ''}`;
      });
    } catch (_) {
      $('eventFeed').innerHTML = '<div class="line"><span style="color:var(--warn)">events unavailable</span></div>';
    }
  }

  async function refreshAttendance() {
    try {
      const out = await getJSON('/api/attendance');
      renderList('attendFeed', out.records || [], 'no attendance records today', (r) => {
        const inTime = (r.clock_in || '-').slice(11, 16) || '-';
        const outTime = (r.clock_out || '-').slice(11, 16) || '-';
        const late = r.late_minutes ? ` · <span style="color:var(--warn)">${r.late_minutes}m late</span>` : '';
        return `<span class="e">${r.name}</span> · in ${inTime} · out ${outTime}${late}`;
      });
    } catch (_) {
      $('attendFeed').innerHTML = '<div class="line"><span style="color:var(--warn)">attendance unavailable</span></div>';
    }
  }

  async function refreshPeople() {
    try {
      const out = await getJSON('/api/people');
      people = out.people || [];
      renderList('peopleFeed', people, 'no enrolled people', (p) => `<span class="e">${p.name}</span>${p.role ? ` · ${p.role}` : ''}`);
      const select = $('peopleSelect');
      if (select) {
        const current = select.value;
        select.innerHTML = people.map((p) => `<option value="${String(p.name).replace(/"/g, '&quot;')}">${p.name}</option>`).join('');
        if (current) select.value = current;
      }
    } catch (_) {
      $('peopleFeed').innerHTML = '<div class="line"><span style="color:var(--warn)">people unavailable</span></div>';
    }
  }

  async function enrollVisible() {
    const name = $('enrollName').value.trim();
    if (!name) {
      setMsg('enter a name first', false);
      return;
    }
    setMsg('enrolling visible unknown...', true);
    try {
      const out = await postJSON('/api/enroll_visible_face', { name });
      setMsg(out.message || 'enrolled', true);
      $('enrollName').value = '';
      await refreshPeople();
    } catch (err) {
      setMsg(String(err.message || err), false);
    }
  }

  async function manualAttendance(kind) {
    const name = $('peopleSelect').value;
    if (!name) {
      setMsg('select a person first', false);
      return;
    }
    try {
      const url = kind === 'in' ? '/api/manual_clock_in' : '/api/manual_clock_out';
      const out = await postJSON(url, { name });
      setMsg(`${name}: ${JSON.stringify(out.result)}`, true);
      await refreshAttendance();
    } catch (err) {
      setMsg(String(err.message || err), false);
    }
  }

  async function askAssistant() {
    const q = $('assistantQuestion').value.trim();
    if (!q) return;
    $('assistantAnswer').textContent = 'thinking...';
    try {
      const out = await postJSON('/api/assistant/ask', { question: q });
      $('assistantAnswer').textContent = out.answer || 'No answer.';
    } catch (err) {
      $('assistantAnswer').textContent = String(err.message || err);
    }
  }

  async function refreshAll() {
    await Promise.all([refreshStats(), refreshLiveState(), refreshEvents(), refreshAttendance()]);
  }

  $('enrollBtn')?.addEventListener('click', enrollVisible);
  $('clockInBtn')?.addEventListener('click', () => manualAttendance('in'));
  $('clockOutBtn')?.addEventListener('click', () => manualAttendance('out'));
  $('askBtn')?.addEventListener('click', askAssistant);
  $('assistantQuestion')?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') askAssistant();
  });

  refreshPeople();
  refreshAll();
  setInterval(refreshAll, 3500);
  setInterval(refreshPeople, 12000);
})();