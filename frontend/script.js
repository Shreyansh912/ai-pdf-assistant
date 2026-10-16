document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadStatus = document.getElementById("uploadStatus");
  const fileNameDisplay = document.getElementById("fileName");
  const pageCountDisplay = document.getElementById("pageCount");
  const chunkCountDisplay = document.getElementById("chunkCount");
  const chatForm = document.getElementById("chatForm");
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesContainer = document.getElementById("messagesContainer");

  let selectedFile = null;

  // Drag & drop handlers
  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = "#6366f1";
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = "#374151";
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".pdf")) {
        fileInput.files = files;
        handleFileSelect(files[0]);
      } else {
        alert("Please drop a valid PDF file.");
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    selectedFile = file;
    if (uploadStatus) uploadStatus.textContent = `Selected: ${file.name}`;
    if (uploadBtn) uploadBtn.disabled = false;
  }

  // Upload handler
  if (uploadBtn) {
    uploadBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        alert("Please select a PDF file first.");
        return;
      }

      const formData = new FormData();
      formData.append("file", selectedFile);

      uploadBtn.disabled = true;
      if (uploadStatus) uploadStatus.textContent = "Uploading & processing...";

      try {
        console.log("Sending upload request...");
        const res = await fetch("/upload", {
          method: "POST",
          body: formData,
        });

        console.log("Upload response status:", res.status);
        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        console.log("Upload success:", data);
        if (uploadStatus) uploadStatus.textContent = "Ready";
        if (fileNameDisplay) fileNameDisplay.textContent = data.filename || selectedFile.name;
        if (pageCountDisplay) pageCountDisplay.textContent = data.page_count ?? "-";
        if (chunkCountDisplay) chunkCountDisplay.textContent = data.chunk_count ?? "-";

        if (userInput) userInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;

        appendMessage("system", `Document "${selectedFile.name}" indexed successfully! Ask anything about it.`);
      } catch (err) {
        console.error("Upload error:", err);
        if (uploadStatus) uploadStatus.textContent = "Error processing document";
        alert(`Upload failed: ${err.message}`);
        uploadBtn.disabled = false;
      }
    });
  }

  // Chat message submission
  if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const question = userInput.value.trim();
      if (!question) return;

      appendMessage("user", question);
      userInput.value = "";
      userInput.disabled = true;
      if (sendBtn) sendBtn.disabled = true;

      const botMessageEl = appendMessage("assistant", "Thinking...");

      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });

        if (!res.ok) throw new Error("Failed to get answer");

        const data = await res.json();
        botMessageEl.querySelector(".bubble").textContent = data.answer || data.response || "No response received.";
      } catch (err) {
        botMessageEl.querySelector(".bubble").textContent = `Error: ${err.message}`;
      } finally {
        userInput.disabled = false;
        if (sendBtn) sendBtn.disabled = false;
        userInput.focus();
      }
    });
  }

  function appendMessage(sender, text) {
    if (!messagesContainer) return null;

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}-message`;

    const avatar = document.createElement("span");
    avatar.className = "avatar";
    avatar.textContent = sender === "user" ? "👤" : sender === "assistant" ? "🤖" : "ℹ️";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return msgDiv;
  }
});