const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const conversation = $("#conversation");
const welcome = $("#welcome");
const messages = $("#messages");
const form = $("#question-form");
const questionInput = $("#question");
const sendButton = $("#send-button");
const topK = $("#top-k");
const uploadDialog = $("#upload-dialog");
const fileInput = $("#paper-files");
const fileSummary = $("#file-summary");
const uploadButton = $("#upload-button");
const uploadProgress = $("#upload-progress");
const dropZone = $("#drop-zone");
let selectedFiles = [];
let toastTimer;

function escapeHTML(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

function resizeTextarea() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 150)}px`;
}

function addUserMessage(question) {
  const message = document.createElement("div");
  message.className = "message user-message";
  message.textContent = question;
  messages.append(message);
}

function addThinkingMessage() {
  const message = document.createElement("div");
  message.className = "message assistant-message";
  message.dataset.thinking = "true";
  message.innerHTML = `
    <div class="assistant-avatar">R</div>
    <div class="assistant-content thinking">
      <p class="assistant-label">ResearchGraph</p>
      <div class="thinking-line"></div>
      <div class="thinking-line short"></div>
      <p class="thinking-status">Searching semantic, lexical, and graph indexes…</p>
    </div>`;
  messages.append(message);
  const stages = [
    "Searching semantic, lexical, and graph indexes…",
    "Fusing evidence across retrieval methods…",
    "Reranking the strongest passages…",
    "Composing a grounded answer…",
  ];
  let stage = 0;
  const timer = setInterval(() => {
    stage = Math.min(stage + 1, stages.length - 1);
    const status = message.querySelector(".thinking-status");
    if (status) status.textContent = stages[stage];
  }, 2400);
  message.dataset.timer = String(timer);
  return message;
}

function renderCitedAnswer(text) {
  // Citation-shaped text inside an equation (for example, x[1]) is math, not a source link.
  const math = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$\n]+?\$)/g;
  return String(text).split(math).map((segment) => {
    const escaped = escapeHTML(segment);
    const isMath = segment.startsWith("$") || segment.startsWith("\\[") || segment.startsWith("\\(");
    return isMath
      ? escaped
      : escaped.replace(/\[(\d+)\]/g, '<button class="citation" type="button" data-citation="$1">[$1]</button>');
  }).join("");
}

async function typesetMath(container) {
  if (!container || !window.MathJax?.startup?.promise) return;
  try {
    await window.MathJax.startup.promise;
    await window.MathJax.typesetPromise([container]);
  } catch (error) {
    console.warn("Math rendering failed; showing the original LaTeX.", error);
  }
}

function methodBadges(methods) {
  return methods
    .map((method) => `<span class="badge ${escapeHTML(method)}">${escapeHTML(method)}</span>`)
    .join("");
}

function renderSource(source) {
  const page = source.page == null ? "page unavailable" : `page ${source.page}`;
  const graphPath = source.graph_path?.length
    ? `<p class="graph-path">Graph path · ${source.graph_path.map(escapeHTML).join(" → ")}</p>`
    : "";
  return `
    <details class="source-card" id="source-${source.index}">
      <summary class="source-summary">
        <span class="source-index">${source.index}</span>
        <span class="source-title">
          <strong>${escapeHTML(source.source)}</strong>
          <small>${page} · chunk ${source.chunk_index}</small>
        </span>
        <span class="source-score">${Math.round(source.rerank_score)} / 100</span>
      </summary>
      <div class="source-body">
        <div class="badges">${methodBadges(source.retrieval_methods)}</div>
        <p>${escapeHTML(source.excerpt)}</p>
        ${graphPath}
      </div>
    </details>`;
}

function renderAssistant(data, thinkingMessage) {
  clearInterval(Number(thinkingMessage.dataset.timer));
  const counts = data.retriever_counts || {};
  const totalUsage = data.usage?.total || {};
  const sourceText = `${data.sources.length} reranked source${data.sources.length === 1 ? "" : "s"}`;
  const retrievalText = `${counts.vector || 0} vector · ${counts.fulltext || 0} lexical · ${counts.graph || 0} graph`;
  const costText = `$${Number(totalUsage.cost_usd || 0).toFixed(4)} query cost`;
  const warning = Object.keys(data.errors || {}).length
    ? `<span class="meta-chip warning">${Object.keys(data.errors).map(escapeHTML).join(", ")} degraded</span>`
    : "";

  thinkingMessage.innerHTML = `
    <div class="assistant-avatar">R</div>
    <div class="assistant-content">
      <p class="assistant-label">ResearchGraph synthesis</p>
      <div class="answer-text">${renderCitedAnswer(data.answer)}</div>
      <div class="answer-meta">
        <span class="meta-chip">${sourceText}</span>
        <span class="meta-chip">${retrievalText}</span>
        <span class="meta-chip">${costText}</span>
        ${warning}
      </div>
      ${data.sources.length ? `
        <div class="sources-heading"><h3>Evidence</h3><span>Ordered by reranker score</span></div>
        <div class="source-list">${data.sources.map(renderSource).join("")}</div>` : ""}
    </div>`;

  thinkingMessage.querySelectorAll(".citation").forEach((citation) => {
    citation.addEventListener("click", () => {
      const source = $(`#source-${citation.dataset.citation}`);
      if (!source) return;
      source.open = true;
      source.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
  void typesetMath(thinkingMessage.querySelector(".answer-text"));
}

function renderError(error, thinkingMessage) {
  clearInterval(Number(thinkingMessage.dataset.timer));
  thinkingMessage.innerHTML = `
    <div class="assistant-avatar">!</div>
    <div class="assistant-content">
      <p class="assistant-label">Could not complete the inquiry</p>
      <div class="answer-text">${escapeHTML(error)}</div>
      <div class="answer-meta"><span class="meta-chip warning">Check the API and server logs</span></div>
    </div>`;
}

async function askQuestion(question) {
  welcome.hidden = true;
  addUserMessage(question);
  const thinkingMessage = addThinkingMessage();
  sendButton.disabled = true;
  questionInput.disabled = true;
  requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
  try {
    const response = await fetch("/hybrid/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: Number(topK.value) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The hybrid query failed.");
    renderAssistant(data, thinkingMessage);
  } catch (error) {
    renderError(error.message || "The hybrid query failed.", thinkingMessage);
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sendButton.disabled) return;
  questionInput.value = "";
  resizeTextarea();
  askQuestion(question);
});

questionInput.addEventListener("input", resizeTextarea);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

$$('.starter').forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.textContent.trim();
    resizeTextarea();
    questionInput.focus();
  });
});

$("#new-chat").addEventListener("click", () => {
  messages.replaceChildren();
  welcome.hidden = false;
  questionInput.value = "";
  resizeTextarea();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

$("#mobile-menu").addEventListener("click", () => document.body.classList.toggle("nav-open"));

function openUpload() {
  uploadDialog.showModal();
}
$("#open-upload").addEventListener("click", openUpload);
$("#top-upload").addEventListener("click", openUpload);

function setFiles(files) {
  selectedFiles = [...files].filter((file) => /\.(pdf|md|txt)$/i.test(file.name));
  fileSummary.textContent = selectedFiles.length
    ? `${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"}: ${selectedFiles.map((file) => file.name).join(", ")}`
    : "No supported files selected";
  uploadButton.disabled = selectedFiles.length === 0;
}

fileInput.addEventListener("change", () => setFiles(fileInput.files));
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", (event) => setFiles(event.dataTransfer?.files || []));

uploadButton.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  const body = new FormData();
  selectedFiles.forEach((file) => body.append("files", file));
  uploadButton.disabled = true;
  uploadProgress.hidden = false;
  try {
    const response = await fetch("/ingest", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed");
    uploadDialog.close();
    showToast(`Indexed ${data.files.length} paper${data.files.length === 1 ? "" : "s"} · ${data.total_chunks} passages`);
    selectedFiles = [];
    fileInput.value = "";
    fileSummary.textContent = "No files selected";
  } catch (error) {
    showToast(error.message || "Upload failed");
  } finally {
    uploadProgress.hidden = true;
    uploadButton.disabled = selectedFiles.length === 0;
  }
});
