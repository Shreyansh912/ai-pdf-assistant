document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadBtn = document.getElementById("upload-btn");
  const statusCard = document.getElementById("status-card");
  const statusMessage = document.getElementById("status-message");
  const statusSpinner = document.getElementById("status-spinner");
  const docMeta = document.getElementById("doc-meta");
  const metaFilename = document.getElementById("meta-filename");
  const metaPages = document.getElementById("meta-pages");
  const metaChunks = document.getElementById("meta-chunks");

  const chatMessages = document.getElementById("chat-messages");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const sendBtn = document.getElementById("send-btn");
  const chatStatus = document.getElementById("chat-status");

  let selectedFile = null;

  ["dragenter", "dragover"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileSelection(fileInput.files[0]);
    }
  });

  function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showStatus("Only PDF files are supported.", "error");
      uploadBtn.disabled = true;
      selectedFile = null;
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      showStatus("File exceeds maximum allowed size of 20MB.", "error");
      uploadBtn.disabled = true;
      selectedFile = null;
      return;
    }

    selectedFile = file;
    showStatus(`Selected: ${file.name}`, "info");
    uploadBtn.disabled = false;
  }

  function showStatus(message, type = "info", showLoading = false) {
    statusCard.classList.remove("hidden");
    statusMessage.textContent = message;
    statusSpinner.classList.toggle("hidden", !showLoading);

    if (type === "error") {
      statusMessage.style.color = "var(--danger)";
    } else if (type === "success") {
      statusMessage.style.color = "var(--success)";
    } else {
      statusMessage.style.color = "var(--text-main)";
    }
  }

  uploadBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    showStatus("Extracting text, chunking, & embedding into FAISS...", "info", true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      showStatus("PDF processed successfully!", "success", false);
      docMeta.classList.remove("hidden");
      metaFilename.textContent = data.filename;
      metaPages.textContent = data.pages;
      metaChunks.textContent = data.chunks;

      questionInput.disabled = false;
      sendBtn.disabled = false;
      chatStatus.textContent = `Active Document: ${data.filename}`;
      questionInput.focus();

    } catch (err) {
      showStatus(err.message, "error", false);
      uploadBtn.disabled = false;
    }
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    appendMessage(question, "user");
    questionInput.value = "";
    questionInput.disabled = true;
    sendBtn.disabled = true;

    const loadingId = appendMessage("Searching document & generating answer...", "assistant", true);

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      const data = await res.json();
      document.getElementById(loadingId)?.remove();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to retrieve answer");
      }

      appendMessage(data.answer, "assistant", false, data.sources);

    } catch (err) {
      document.getElementById(loadingId)?.remove();
      appendMessage(`Error: ${err.message}`, "assistant");
    } finally {
      questionInput.disabled = false;
      sendBtn.disabled = false;
      questionInput.focus();
    }
  });

  function appendMessage(text, sender, isLoading = false, sources = []) {
    const msgId = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.id = msgId;
    msgDiv.classList.add("message", sender === "user" ? "user-msg" : "assistant-msg");

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = sender === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (isLoading) {
      bubble.style.fontStyle = "italic";
      bubble.style.color = "var(--text-muted)";
    }

    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);

    if (sources && sources.length > 0) {
      const sourcesBox = document.createElement("div");
      sourcesBox.className = "sources-box";

      const sourcesTitle = document.createElement("div");
      sourcesTitle.className = "sources-title";
      sourcesTitle.textContent = "Sources";
      sourcesBox.appendChild(sourcesTitle);

      sources.forEach(src => {
        const badge = document.createElement("span");
        badge.className = "source-badge";
        badge.textContent = `${src.filename} — Page ${src.page}`;
        sourcesBox.appendChild(badge);
      });

      bubble.appendChild(sourcesBox);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;
    return msgId;
  }
});
