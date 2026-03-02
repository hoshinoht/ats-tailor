import { EditorView, lineNumbers, highlightActiveLine, highlightSpecialChars, drawSelection, rectangularSelection, keymap } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { yaml } from '@codemirror/lang-yaml';
import { oneDark } from '@codemirror/theme-one-dark';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, indentOnInput, foldGutter, foldKeymap } from '@codemirror/language';
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';

let view = null;
let updateCallback = null;

function makeExtensions(dark) {
  return [
    lineNumbers(),
    highlightActiveLine(),
    highlightSpecialChars(),
    drawSelection(),
    rectangularSelection(),
    history(),
    indentOnInput(),
    bracketMatching(),
    closeBrackets(),
    foldGutter(),
    highlightSelectionMatches(),
    keymap.of([
      ...defaultKeymap,
      ...historyKeymap,
      ...foldKeymap,
      ...searchKeymap,
      ...closeBracketsKeymap,
      indentWithTab,
    ]),
    yaml(),
    ...(dark ? [oneDark] : []),
    EditorView.updateListener.of((update) => {
      if (update.docChanged && updateCallback) {
        updateCallback(update.state.doc.toString());
      }
    }),
  ];
}

export function initRawEditor(container) {
  const state = EditorState.create({ doc: '', extensions: makeExtensions(false) });
  view = new EditorView({ state, parent: container });
  return view;
}

export function applyDarkMode(dark) {
  if (!view) return;
  const content = view.state.doc.toString();
  const parent = view.dom.parentElement;
  view.destroy();
  const state = EditorState.create({ doc: content, extensions: makeExtensions(dark) });
  view = new EditorView({ state, parent });
}

export function setContent(text) {
  if (!view) return;
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: text || '' },
  });
}

export function getContent() {
  if (!view) return '';
  return view.state.doc.toString();
}

export function onUpdate(callback) {
  updateCallback = callback;
}

export function setErrors(errors) {
  const container = view ? view.dom.parentElement : null;
  if (!container) return;
  let banner = container.querySelector('.raw-error-banner');
  if (!errors || errors.length === 0) {
    if (banner) banner.remove();
    return;
  }
  if (!banner) {
    banner = document.createElement('div');
    banner.className = 'raw-error-banner';
    banner.style.cssText = 'background:#ff3b30;color:#fff;padding:6px 12px;font-size:12px;';
    container.insertBefore(banner, container.firstChild);
  }
  banner.textContent = errors.map(e => `Line ${e.line}: ${e.message}`).join(' | ');
}
