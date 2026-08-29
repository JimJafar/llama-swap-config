/* EnGram Explorer frontend */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  seq: [],           // token ids in the current sequence
  seqText: [],       // decoded text per seq position
};

const status = {
  el: $("statusDot"), tx: $("statusText"),
  set(mode, label) {
    this.el.className = "dot " + mode;
    this.tx.textContent = label;
  },
};

/* ---------- token search dropdown ---------- */
const search = $("tokenSearch");
const dropdown = $("tokenDropdown");
let ddItems = [];
let ddSel = 0;

async function doSearch(q) {
  if (!q) { dropdown.classList.add("hidden"); return; }
  status.set("busy", "searching vocab…");
  let data;
  try {
    const res = await fetch(`/search?q=${encodeURIComponent(q)}&limit=50`);
    data = await res.json();
  } catch (e) {
    status.set("err", "search failed"); return;
  }
  ddItems = data.results || [];
  ddSel = 0;
  renderDropdown();
  status.set("ok", `${ddItems.length} match${ddItems.length === 1 ? "" : "es"}`);
}

function renderDropdown() {
  if (!ddItems.length) {
    dropdown.innerHTML = `<div class="dd-empty">no tokens match</div>`;
    dropdown.classList.remove("hidden");
    return;
  }
  dropdown.innerHTML = "";
  ddItems.forEach((r, i) => {
    const el = document.createElement("div");
    el.className = "dd-item" + (i === ddSel ? " sel" : "");
    el.innerHTML = `<span class="dd-text"></span><span class="dd-id">#${r.id}</span><span class="dd-tag">${r.match === "prefix" ? "prefix" : "sub"}</span>`;
    el.querySelector(".dd-text").textContent = r.text;
    el.addEventListener("mousedown", (ev) => { ev.preventDefault(); pickToken(r); });
    el.addEventListener("mouseenter", () => { ddSel = i; renderDropdown(); });
    dropdown.appendChild(el);
  });
  dropdown.classList.remove("hidden");
}

let debounce;
search.addEventListener("input", () => {
  clearTimeout(debounce);
  debounce = setTimeout(() => doSearch(search.value), 120);
});
search.addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowDown") { ddSel = (ddSel + 1) % ddItems.length; renderDropdown(); ev.preventDefault(); }
  else if (ev.key === "ArrowUp") { ddSel = (ddSel - 1 + ddItems.length) % ddItems.length; renderDropdown(); ev.preventDefault(); }
  else if (ev.key === "Enter") { if (ddItems[ddSel]) pickToken(ddItems[ddSel]); }
  else if (ev.key === "Escape") { dropdown.classList.add("hidden"); }
});
document.addEventListener("click", (ev) => {
  if (!ev.target.closest(".seed")) dropdown.classList.add("hidden");
});

function pickToken(r) {
  if (r.id < 0) return;
  addToken(r.id, r.text);
  search.value = "";
  dropdown.classList.add("hidden");
}

/* ---------- encode & start ---------- */
$("seedBtn").addEventListener("click", async () => {
  const text = search.value.trim();
  if (!text) return;
  status.set("busy", "encoding…");
  let data;
  try {
    const res = await fetch(`/encode?text=${encodeURIComponent(text)}`);
    data = await res.json();
  } catch (e) { status.set("err", "encode failed"); return; }
  state.seq = []; state.seqText = [];
  for (const t of data.tokens) { state.seq.push(t.id); state.seqText.push(t.text); }
  search.value = "";
  dropdown.classList.add("hidden");
  renderSequence();
  status.set("busy", "querying continuations…");
  await fetchContinuations();
});

/* ---------- sequence ---------- */
function renderSequence() {
  const bar = $("sequenceBar");
  bar.classList.toggle("empty", state.seq.length === 0);
  bar.innerHTML = "";
  if (!state.seq.length) {
    bar.innerHTML = `<span class="placeholder">Select a token below or press <em>Encode &amp; start</em>.</span>`;
    return;
  }
  state.seq.forEach((id, i) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span class="ct">${escapeHtml(state.seqText[i])}</span><span class="cid">#${id}</span><span class="x">×</span>`;
    chip.querySelector(".x").addEventListener("click", () => { state.seq.splice(i, 1); state.seqText.splice(i, 1); renderSequence(); fetchContinuations(); });
    bar.appendChild(chip);
  });
}

/* ---------- continuations ---------- */
const contEl = $("continuations");
const contLabel = $("contLabel");
const contMeta = $("contMeta");

async function fetchContinuations() {
  const seq = state.seq;
  if (!seq.length) {
    contLabel.textContent = "Next-token continuations";
    contMeta.textContent = "";
    contEl.innerHTML = `<div class="empty-state"><p>Nothing here yet.</p><p class="muted">Build a sequence first.</p></div>`;
    return;
  }
  status.set("busy", "model…");
  const t = seq.join(",");
  let data;
  try {
    const res = await fetch(`/continue?t=${t}&k=12`);
    data = await res.json();
  } catch (e) {
    status.set("err", "model request failed");
    return;
  }
  if (data.error) { status.set("err", data.error); return; }
  renderContinuations(data.continuations || [], seq);
  status.set("ok", "ready");
}

function renderContinuations(list, seq) {
  if (!list.length) {
    contEl.innerHTML = `<div class="empty-state"><p class="muted">No continuations returned.</p></div>`;
    return;
  }
  contLabel.textContent = "Next-token continuations";
  contMeta.textContent = `prefix: ${seq.length} tok · model top-${list.length}`;
  contEl.innerHTML = "";
  list.forEach((c, i) => {
    const el = document.createElement("div");
    el.className = "cont" + (i === 0 ? " top1" : "");
    const pct = c.logprob != null ? c.logprob.toFixed(2) : "–";
    el.innerHTML = `<span class="prob">${pct}</span><span class="ctext">${escapeHtml(c.text)}</span><span class="cid">#${c.id}</span>`;
    el.addEventListener("click", () => {
      addToken(c.id, c.text);
    });
    contEl.appendChild(el);
  });
}

function addToken(id, text) {
  state.seq.push(id);
  state.seqText.push(text);
  renderSequence();
  fetchContinuations();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* initial */
$("clearBtn").addEventListener("click", () => {
  state.seq = []; state.seqText = [];
  renderSequence();
  contEl.innerHTML = `<div class="empty-state"><p>Cleared.</p><p class="muted">Build a new sequence.</p></div>`;
  contLabel.textContent = "Next-token continuations";
  contMeta.textContent = "";
  status.set("ok", "cleared");
});

fetch("/health").then(r => r.json()).then(d => {
  status.set("ok", `vocab ${d.vocab} tokens`);
}).catch(() => status.set("err", "backend down"));
