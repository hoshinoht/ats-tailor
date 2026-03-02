import { renderTree } from './tree.js';
import { initRawEditor, setContent, getContent, onUpdate, setErrors, applyDarkMode } from './raw.js';

import jsyaml from 'js-yaml';

// ===== State =====

export const state = {
  currentFile: null,
  mode: 'tree',       // 'tree' | 'raw'
  files: {},          // { name: { raw, data, mtime } }
  dirty: false,
  darkMode: false,
};

// ===== DOM refs =====

const tabBar = document.getElementById('tabBar');
const treeContainer = document.getElementById('treeContainer');
const rawContainer = document.getElementById('rawContainer');
const statusPill = document.getElementById('statusPill');
const modeToggle = document.getElementById('modeToggle');
const btnTree = document.getElementById('btnTree');
const btnRaw = document.getElementById('btnRaw');
const darkToggle = document.getElementById('darkToggle');
const externalBanner = document.getElementById('externalChangeBanner');
const bannerReload = document.getElementById('bannerReload');
const bannerIgnore = document.getElementById('bannerIgnore');
const searchInput = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');

// ===== Status Pill =====

let statusTimeout = null;

function showStatus(msg, type) {
  statusPill.textContent = msg;
  statusPill.className = 'status-pill ' + type;
  statusPill.classList.remove('hidden');
  if (statusTimeout) clearTimeout(statusTimeout);
  if (type === 'saved') {
    statusTimeout = setTimeout(() => statusPill.classList.add('hidden'), 2000);
  }
}

// ===== Auto-save (debounce 1.5s) =====

let saveTimer = null;

function scheduleSave() {
  state.dirty = true;
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(saveFile, 1500);
}

// ===== File Operations =====

export async function loadFile(name) {
  const res = await fetch('/api/files/' + name);
  if (!res.ok) { showStatus('Error loading ' + name, 'error'); return; }
  const json = await res.json();
  state.files[name] = { raw: json.raw, data: json.data, mtime: json.mtime };
  state.currentFile = name;
  renderCurrentMode();
  updateTabs();
}

export async function saveFile() {
  if (!state.currentFile) return;
  const raw = state.mode === 'raw' ? getContent() : jsyaml.dump(state.files[state.currentFile].data);
  showStatus('Saving\u2026', 'saving');
  try {
    const res = await fetch('/api/files/' + state.currentFile, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ raw }),
    });
    if (!res.ok) throw new Error(await res.text());
    const json = await res.json();
    state.files[state.currentFile].mtime = json.mtime;
    state.files[state.currentFile].raw = raw;
    state.dirty = false;
    showStatus('Saved \u2713', 'saved');
  } catch (e) {
    showStatus('Error: ' + e.message, 'error');
  }
}

// ===== Mode Rendering =====

function renderCurrentMode() {
  if (!state.currentFile) return;
  const fileState = state.files[state.currentFile];
  if (state.mode === 'tree') {
    treeContainer.classList.remove('hidden');
    rawContainer.classList.add('hidden');
    renderTree(treeContainer, state.currentFile, fileState.data, (updated) => {
      fileState.data = updated;
      scheduleSave();
    });
  } else {
    treeContainer.classList.add('hidden');
    rawContainer.classList.remove('hidden');
    setContent(fileState.raw || '');
  }
}

// ===== Mode Toggle =====

function setMode(mode) {
  if (mode === state.mode) return;

  if (mode === 'raw' && state.mode === 'tree') {
    // Tree -> Raw: serialize current data
    if (state.currentFile && state.files[state.currentFile]) {
      const raw = jsyaml.dump(state.files[state.currentFile].data);
      state.files[state.currentFile].raw = raw;
    }
  } else if (mode === 'tree' && state.mode === 'raw') {
    // Raw -> Tree: parse and validate
    const raw = getContent();
    try {
      const data = jsyaml.load(raw);
      if (state.currentFile && state.files[state.currentFile]) {
        state.files[state.currentFile].data = data;
        state.files[state.currentFile].raw = raw;
      }
      setErrors([]);
    } catch (e) {
      const lineMatch = e.mark ? [{ line: e.mark.line + 1, message: e.message }] : [];
      setErrors(lineMatch);
      alert('Cannot switch to Tree mode: YAML parse error\n\n' + e.message + '\n\nFix the error in Raw mode first.');
      return;
    }
  }

  state.mode = mode;
  btnTree.classList.toggle('active', mode === 'tree');
  btnRaw.classList.toggle('active', mode === 'raw');
  renderCurrentMode();
}

btnTree.onclick = () => {
  if (state.mode === 'raw') {
    const confirmed = window.confirm('Switching to tree mode will lose YAML comments. Continue?');
    if (!confirmed) return;
  }
  setMode('tree');
};
btnRaw.onclick = () => setMode('raw');

// ===== Tabs =====

function buildTabs(files) {
  while (tabBar.firstChild) tabBar.removeChild(tabBar.firstChild);
  files.forEach(name => {
    const tab = document.createElement('button');
    tab.className = 'tab' + (name === state.currentFile ? ' active' : '');
    tab.textContent = name.replace('.yaml', '');
    tab.onclick = async () => {
      if (name === state.currentFile) return;
      if (state.dirty) await saveFile();
      await loadFile(name);
    };
    tabBar.appendChild(tab);
  });
}

function updateTabs() {
  Array.from(tabBar.querySelectorAll('.tab')).forEach(tab => {
    tab.classList.toggle('active', tab.textContent === state.currentFile);
  });
}

// ===== Dark Mode =====

function applyTheme(dark) {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  applyDarkMode(dark);
}

darkToggle.onclick = () => {
  state.darkMode = !state.darkMode;
  localStorage.setItem('yamledit-dark', state.darkMode ? '1' : '0');
  applyTheme(state.darkMode);
};

// ===== SSE =====

function connectSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.file && msg.file === state.currentFile) {
        externalBanner.classList.remove('hidden');
      }
    } catch (_) {}
  };
  es.onerror = () => {
    es.close();
    setTimeout(connectSSE, 3000);
  };
}

bannerReload.onclick = async () => {
  externalBanner.classList.add('hidden');
  await loadFile(state.currentFile);
};
bannerIgnore.onclick = () => externalBanner.classList.add('hidden');

// ===== Auto-save from raw editor =====

onUpdate((text) => {
  if (state.mode === 'raw' && state.currentFile) {
    state.files[state.currentFile].raw = text;
    scheduleSave();
  }
});

// ===== Search =====

let searchActiveIdx = -1;

function flattenToSearchable(obj) {
  // Recursively extract all string values from an object for full-text matching
  const parts = [];
  if (obj == null) return '';
  if (typeof obj === 'string') return obj;
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (Array.isArray(obj)) {
    for (const item of obj) parts.push(flattenToSearchable(item));
  } else if (typeof obj === 'object') {
    for (const v of Object.values(obj)) parts.push(flattenToSearchable(v));
  }
  return parts.join(' ');
}

function extractSearchItems(fileName, data) {
  const items = [];
  const name = fileName.replace('.yaml', '');
  if (!data) return items;

  if (name === 'certifications') {
    const list = data.certifications || (Array.isArray(data) ? data : []);
    list.forEach((c, i) => {
      items.push({
        file: fileName,
        title: c.name || `Cert #${i+1}`,
        context: [c.id, c.issuer, ...(c.ats_keywords || []), ...(c.relevance || [])].filter(Boolean).join(' · '),
        searchText: flattenToSearchable(c),
      });
    });
  } else if (name === 'skills') {
    const cats = data.categories || (Array.isArray(data) ? data : []);
    cats.forEach((cat, ci) => {
      (cat.skills || []).forEach((s, si) => {
        const evidenceRefs = (s.evidence || []).map(e => [e.type, e.ref, e.detail].filter(Boolean).join(' ')).join(', ');
        items.push({
          file: fileName,
          title: s.name || `Skill #${si+1}`,
          context: [cat.name, s.proficiency, ...(s.ats_keywords || [])].filter(Boolean).join(' · '),
          searchText: flattenToSearchable(s) + ' ' + (cat.name || ''),
        });
      });
    });
  } else if (name === 'projects') {
    const list = data.projects || (Array.isArray(data) ? data : []);
    list.forEach((p, i) => {
      items.push({
        file: fileName,
        title: p.name || `Project #${i+1}`,
        context: [p.id, p.type, p.period, ...(p.ats_tags || [])].filter(Boolean).join(' · '),
        searchText: flattenToSearchable(p),
      });
    });
  } else if (name === 'experience') {
    (data.roles || []).forEach((r, i) => {
      items.push({
        file: fileName,
        title: `${r.title || 'Role #' + (i+1)}`,
        context: [r.id, r.company, r.type, r.period, ...(r.skills_used || [])].filter(Boolean).join(' · '),
        searchText: flattenToSearchable(r),
      });
    });
    (data.education || []).forEach((e, i) => {
      items.push({
        file: fileName,
        title: `${e.degree || 'Education #' + (i+1)}`,
        context: [e.institution, e.field, e.period].filter(Boolean).join(' · '),
        searchText: flattenToSearchable(e),
      });
    });
  }
  return items;
}

function highlightMatch(text, query) {
  if (!query) return document.createTextNode(text);
  const frag = document.createDocumentFragment();
  const lower = text.toLowerCase();
  const qLower = query.toLowerCase();
  let start = 0;
  let idx;
  while ((idx = lower.indexOf(qLower, start)) !== -1) {
    if (idx > start) frag.appendChild(document.createTextNode(text.slice(start, idx)));
    const mark = document.createElement('mark');
    mark.textContent = text.slice(idx, idx + query.length);
    frag.appendChild(mark);
    start = idx + query.length;
  }
  if (start < text.length) frag.appendChild(document.createTextNode(text.slice(start)));
  return frag;
}

function runSearch(query) {
  searchActiveIdx = -1;
  while (searchResults.firstChild) searchResults.removeChild(searchResults.firstChild);

  if (!query || query.length < 2) {
    searchResults.classList.add('hidden');
    return;
  }

  const qLower = query.toLowerCase();
  const grouped = {};
  let totalHits = 0;

  for (const [fileName, fileState] of Object.entries(state.files)) {
    const items = extractSearchItems(fileName, fileState.data);
    const matches = items.filter(it =>
      it.searchText.toLowerCase().includes(qLower)
    );
    if (matches.length) {
      grouped[fileName] = matches;
      totalHits += matches.length;
    }
  }

  if (totalHits === 0) {
    const empty = document.createElement('div');
    empty.className = 'search-empty';
    empty.textContent = 'No matches for "' + query + '"';
    searchResults.appendChild(empty);
    searchResults.classList.remove('hidden');
    return;
  }

  let globalIdx = 0;
  for (const [fileName, matches] of Object.entries(grouped)) {
    const header = document.createElement('div');
    header.className = 'search-group-header';
    header.textContent = fileName;
    searchResults.appendChild(header);

    for (const match of matches) {
      const item = document.createElement('div');
      item.className = 'search-item';
      item.dataset.idx = globalIdx++;
      item.dataset.file = match.file;

      const titleEl = document.createElement('div');
      titleEl.className = 'search-item-title';
      titleEl.appendChild(highlightMatch(match.title, query));
      item.appendChild(titleEl);

      const ctxEl = document.createElement('div');
      ctxEl.className = 'search-item-context';
      ctxEl.appendChild(highlightMatch(match.context, query));
      item.appendChild(ctxEl);

      item.onclick = async () => {
        searchResults.classList.add('hidden');
        searchInput.value = '';
        if (match.file !== state.currentFile) {
          if (state.dirty) await saveFile();
          await loadFile(match.file);
        }
      };
      searchResults.appendChild(item);
    }
  }
  searchResults.classList.remove('hidden');
}

let searchDebounce = null;
searchInput.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => runSearch(searchInput.value.trim()), 150);
});

searchInput.addEventListener('keydown', (e) => {
  const items = searchResults.querySelectorAll('.search-item');
  if (!items.length) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    searchActiveIdx = Math.min(searchActiveIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle('active', i === searchActiveIdx));
    items[searchActiveIdx]?.scrollIntoView({ block: 'nearest' });
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    searchActiveIdx = Math.max(searchActiveIdx - 1, 0);
    items.forEach((el, i) => el.classList.toggle('active', i === searchActiveIdx));
    items[searchActiveIdx]?.scrollIntoView({ block: 'nearest' });
  } else if (e.key === 'Enter' && searchActiveIdx >= 0) {
    e.preventDefault();
    items[searchActiveIdx]?.click();
  } else if (e.key === 'Escape') {
    searchResults.classList.add('hidden');
    searchInput.blur();
  }
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrap')) {
    searchResults.classList.add('hidden');
  }
});

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    searchInput.focus();
    searchInput.select();
  }
});

// ===== Init =====

export async function init() {
  // Dark mode from localStorage
  const savedDark = localStorage.getItem('yamledit-dark');
  if (savedDark === '1') {
    state.darkMode = true;
    applyTheme(true);
  }

  // Init raw editor
  initRawEditor(rawContainer);

  // Fetch file list
  let files = [];
  try {
    const res = await fetch('/api/files');
    if (res.ok) {
      const json = await res.json();
      files = Array.isArray(json) ? json : (json.files || []);
    }
  } catch (e) {
    showStatus('Error fetching file list', 'error');
    return;
  }

  buildTabs(files);

  // Preload all files for cross-file search
  await Promise.all(files.map(async (name) => {
    const res = await fetch('/api/files/' + name);
    if (res.ok) {
      const json = await res.json();
      state.files[name] = { raw: json.raw, data: json.data, mtime: json.mtime };
    }
  }));

  if (files.length > 0) {
    state.currentFile = files[0];
    renderCurrentMode();
    updateTabs();
  }

  connectSSE();
}

init();
