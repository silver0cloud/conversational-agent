// Profile dashboard — plain fetch + DOM, no framework, matching app.js's style.

const el = {
  tagsContainer: document.getElementById('tags-container'),
  notesText: document.getElementById('notes-text'),
  profileEmpty: document.getElementById('profile-empty'),
  conversationsList: document.getElementById('conversations-list'),
  conversationsEmpty: document.getElementById('conversations-empty'),
  toast: document.getElementById('toast'),
};

const STATUS_LABELS = {
  in_progress: 'Live',
  ended: 'Wrapping up',
  processing: 'Processing',
  done: 'Ready',
  error: 'Error',
};

function formatDate(isoString) {
  if (!isoString) return '';
  // SQLite datetime('now') stores UTC without a timezone suffix — append
  // one so the browser doesn't misinterpret it as local time.
  const date = new Date(isoString.replace(' ', 'T') + 'Z');
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.remove('hidden');
  setTimeout(() => el.toast.classList.add('hidden'), 2000);
}

// ------------------------------------------------------------- profile ---

async function loadProfile() {
  try {
    const res = await fetch('/api/profile');
    const profile = await res.json();

    const tags = profile.tags || [];
    const notes = (profile.notes || '').trim();

    if (tags.length === 0 && !notes) {
      el.profileEmpty.classList.remove('hidden');
      return;
    }

    el.tagsContainer.innerHTML = tags
      .map((tag) => `<span class="tag-pill">${escapeHtml(tag)}</span>`)
      .join('');
    el.notesText.textContent = notes;
  } catch (err) {
    console.error('Failed to load profile:', err);
  }
}

// -------------------------------------------------------- conversations --

async function loadConversations() {
  try {
    const res = await fetch('/api/conversations');
    const conversations = await res.json();

    if (conversations.length === 0) {
      el.conversationsEmpty.classList.remove('hidden');
      return;
    }

    el.conversationsList.innerHTML = conversations.map(renderConversationRow).join('');

    conversations.forEach((conv) => {
      const row = document.getElementById(`conv-${conv.id}`);
      row.addEventListener('click', () => toggleConversation(conv.id));
    });
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
}

function renderConversationRow(conv) {
  const topic = conv.topic || 'Untitled conversation';
  const statusLabel = STATUS_LABELS[conv.status] || conv.status;
  return `
    <div class="conversation-row" id="conv-${conv.id}">
      <div class="conversation-row-top">
        <span class="conversation-topic">${escapeHtml(topic)}</span>
        <span class="conversation-meta">
          <span class="conversation-date">${formatDate(conv.started_at)}</span>
          <span class="status-badge status-${conv.status}">${escapeHtml(statusLabel)}</span>
        </span>
      </div>
      <div class="conversation-detail" id="conv-detail-${conv.id}"></div>
    </div>
  `;
}

const loadedDetails = new Set();

async function toggleConversation(id) {
  const row = document.getElementById(`conv-${id}`);
  row.classList.toggle('expanded');

  if (row.classList.contains('expanded') && !loadedDetails.has(id)) {
    await loadConversationDetail(id);
  }
}

async function loadConversationDetail(id) {
  const detailEl = document.getElementById(`conv-detail-${id}`);
  detailEl.innerHTML = '<p class="detail-loading">Loading…</p>';

  try {
    const res = await fetch(`/api/conversations/${id}`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const conv = await res.json();

    if (conv.status === 'in_progress' || conv.status === 'ended') {
      detailEl.innerHTML = '<p class="detail-loading">Still processing — check back in a moment.</p>';
      return;
    }
    if (conv.status === 'error') {
      detailEl.innerHTML = `<p class="detail-error">Processing failed: ${escapeHtml(conv.error_message || 'unknown error')}</p>`;
      return;
    }

    loadedDetails.add(id);
    detailEl.innerHTML = `
      <p class="detail-label">Summary</p>
      <p class="detail-summary">${escapeHtml(conv.summary || '(no summary)')}</p>
      <div class="detail-actions">
        <button class="action-btn" data-action="copy-summary">Copy summary</button>
        <a class="action-btn" href="/api/conversations/${id}/draft.md" download>Download Substack draft (.md)</a>
        <button class="action-btn" data-action="copy-draft">Copy draft</button>
      </div>
    `;

    detailEl.querySelector('[data-action="copy-summary"]').addEventListener('click', (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(conv.summary || '');
      showToast('Summary copied');
    });
    detailEl.querySelector('[data-action="copy-draft"]').addEventListener('click', (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(conv.substack_draft || '');
      showToast('Draft copied');
    });
  } catch (err) {
    console.error('Failed to load conversation detail:', err);
    detailEl.innerHTML = '<p class="detail-error">Failed to load details — try refreshing.</p>';
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// -------------------------------------------------------------- start ----

loadProfile();
loadConversations();
