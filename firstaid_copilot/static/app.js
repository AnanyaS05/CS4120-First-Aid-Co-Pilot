const state = {
  conversations: [],
  currentSessionId: null,
  currentTitle: "New Chat",
  messages: [],
  sending: false,
  pendingAssistantId: null,
  lastSubmittedQuery: "",
};

const STATUS_LABELS = {
  started: "Starting response…",
  retrieving: "Retrieving context…",
  retrying: "Retrying response…",
  fallback: "Using fallback guidance…",
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  bindEvents();
  autoResizeTextarea();
  initializeApp().catch((error) => {
    console.error(error);
    setStatus("Failed to load the chat UI.");
  });
});

function cacheDom() {
  dom.chatTitle = document.getElementById("chat-title");
  dom.conversationList = document.getElementById("conversation-list");
  dom.emptyState = document.getElementById("empty-state");
  dom.messageInput = document.getElementById("message-input");
  dom.messages = document.getElementById("messages");
  dom.messageTemplate = document.getElementById("message-template");
  dom.conversationTemplate = document.getElementById("conversation-item-template");
  dom.modelSelect = document.getElementById("model-select");
  dom.newChatButton = document.getElementById("new-chat-button");
  dom.profileSelect = document.getElementById("profile-select");
  dom.sendButton = document.getElementById("send-button");
  dom.statusPill = document.getElementById("status-pill");
}

function bindEvents() {
  dom.newChatButton.addEventListener("click", startNewChat);
  dom.sendButton.addEventListener("click", () => {
    void sendCurrentMessage();
  });
  dom.messageInput.addEventListener("input", autoResizeTextarea);
  dom.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendCurrentMessage();
    }
  });
  dom.conversationList.addEventListener("click", (event) => {
    const button = event.target.closest(".conversation-item");
    if (!button) {
      return;
    }
    const { sessionId } = button.dataset;
    if (!sessionId) {
      return;
    }
    void openConversation(sessionId);
  });
}

async function initializeApp() {
  await Promise.all([loadModels(), refreshConversationList()]);
  if (state.conversations.length > 0) {
    await openConversation(state.conversations[0].session_id);
  } else {
    renderConversationList();
    renderMessages();
  }
}

async function loadModels() {
  const response = await fetch("/models");
  if (!response.ok) {
    populateModels([
      { name: "functiongemma", available: false },
      { name: "qwen3:0.6b", available: true },
      { name: "qwen3.5:0.8b", available: true },
      { name: "granite4:350m", available: false },
    ]);
    return;
  }

  populateModels(await response.json());
}

function populateModels(models) {
  dom.modelSelect.replaceChildren();
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.available ? model.name : `${model.name} (offline)`;
    option.disabled = !model.available;
    dom.modelSelect.append(option);
  });

  const firstEnabled = Array.from(dom.modelSelect.options).find((option) => !option.disabled);
  if (firstEnabled) {
    dom.modelSelect.value = firstEnabled.value;
  }
}

async function refreshConversationList() {
  const response = await fetch("/conversations");
  if (!response.ok) {
    throw new Error("Failed to load conversations.");
  }
  state.conversations = await response.json();
  renderConversationList();
}

async function openConversation(sessionId) {
  const response = await fetch(`/conversations/${encodeURIComponent(sessionId)}`);
  if (!response.ok) {
    throw new Error("Failed to load the selected conversation.");
  }

  const thread = await response.json();
  state.currentSessionId = thread.session_id;
  state.currentTitle = thread.title;
  state.messages = thread.messages.map(normalizeMessage);
  state.pendingAssistantId = null;

  const summary = state.conversations.find((item) => item.session_id === sessionId);
  if (summary) {
    if (summary.model && findOption(dom.modelSelect, summary.model)) {
      dom.modelSelect.value = summary.model;
    }
    if (summary.profile) {
      dom.profileSelect.value = summary.profile;
    }
  }

  renderConversationList();
  renderMessages();
  scrollMessagesToBottom(false);
}

function startNewChat() {
  state.currentSessionId = null;
  state.currentTitle = "New Chat";
  state.messages = [];
  state.pendingAssistantId = null;
  state.lastSubmittedQuery = "";
  renderConversationList();
  renderMessages();
  setStatus("");
  dom.messageInput.focus();
}

async function sendCurrentMessage() {
  const text = dom.messageInput.value.trim();
  if (!text || state.sending) {
    return;
  }

  state.sending = true;
  state.lastSubmittedQuery = text;
  dom.sendButton.disabled = true;
  dom.messageInput.value = "";
  autoResizeTextarea();

  const timestamp = new Date().toISOString();
  appendMessage({
    id: makeClientId("user"),
    role: "user",
    text,
    timestamp,
  });

  const assistantId = makeClientId("assistant");
  state.pendingAssistantId = assistantId;
  appendMessage({
    id: assistantId,
    role: "assistant",
    text: "",
    timestamp,
    sources: [],
    warnings: [],
    streaming: true,
  });

  const payload = {
    query: text,
    model: dom.modelSelect.value,
    profile: dom.profileSelect.value,
    top_k: 5,
  };
  if (state.currentSessionId) {
    payload.session_id = state.currentSessionId;
  }

  try {
    await streamChat(payload);
  } catch (error) {
    console.error(error);
    updateAssistantMessage({
      text: "The request failed before a complete response was received.",
      warnings: ["The UI could not complete the streaming request."],
      streaming: false,
    });
    setStatus("Request failed.");
  } finally {
    state.sending = false;
    dom.sendButton.disabled = false;
  }
}

async function streamChat(payload) {
  const response = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Streaming request failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const rawEvent = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);
      const parsed = parseSseEvent(rawEvent);
      if (parsed) {
        handleStreamEvent(parsed.type, parsed.data);
      }
      boundaryIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}

function parseSseEvent(rawEvent) {
  if (!rawEvent.trim()) {
    return null;
  }

  let type = "message";
  const dataLines = [];
  rawEvent.split(/\r?\n/).forEach((line) => {
    if (line.startsWith("event:")) {
      type = line.slice("event:".length).trim();
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  });

  if (!dataLines.length) {
    return null;
  }

  return {
    type,
    data: JSON.parse(dataLines.join("\n")),
  };
}

function handleStreamEvent(type, data) {
  if (type === "session") {
    state.currentSessionId = data.session_id;
    state.currentTitle = truncateText(state.lastSubmittedQuery || "New Chat", 56);
    upsertConversationSummary({
      session_id: data.session_id,
      title: state.currentTitle,
      preview: "Streaming response…",
      updated_at: new Date().toISOString(),
      turn_count: Math.max(1, Math.ceil(state.messages.length / 2)),
      model: data.model,
      profile: data.profile,
    });
    renderConversationList();
    renderTitle();
    return;
  }

  if (type === "status") {
    setStatus(STATUS_LABELS[data.value] || "");
    return;
  }

  if (type === "retrieval") {
    updateAssistantMessage({ sources: data.hits || [] });
    return;
  }

  if (type === "token") {
    const current = getPendingAssistantMessage();
    const nextText = `${current?.text || ""}${data.text || ""}`;
    updateAssistantMessage({ text: nextText, streaming: true });
    return;
  }

  if (type === "warning") {
    const currentWarnings = getPendingAssistantMessage()?.warnings || [];
    updateAssistantMessage({
      warnings: [...currentWarnings, data.message].filter(Boolean),
    });
    return;
  }

  if (type === "final") {
    state.currentSessionId = data.session_id;
    state.currentTitle = truncateText(
      state.lastSubmittedQuery || state.currentTitle || "New Chat",
      56,
    );
    updateAssistantMessage({
      text: data.answer_text,
      sources: data.sources || [],
      warnings: data.warnings || [],
      streaming: false,
    });
    setStatus("");
    void refreshConversationList();
    renderConversationList();
    return;
  }

  if (type === "error") {
    updateAssistantMessage({
      warnings: [data.message || "Streaming failed."],
      streaming: false,
    });
    setStatus("Streaming failed.");
  }
}

function appendMessage(message) {
  const normalized = normalizeMessage(message);
  state.messages.push(normalized);
  const element = createMessageElement(normalized);
  dom.messages.append(element);
  toggleEmptyState();
  scrollMessagesToBottom(true);
}

function updateAssistantMessage(patch) {
  const message = getPendingAssistantMessage();
  if (!message) {
    return;
  }

  Object.assign(message, patch);
  const element = dom.messages.querySelector(
    `[data-message-id="${cssEscape(message.id)}"]`,
  );
  if (element) {
    hydrateMessageElement(element, message);
  }
  scrollMessagesToBottom(true);
}

function getPendingAssistantMessage() {
  return state.messages.find((message) => message.id === state.pendingAssistantId);
}

function normalizeMessage(message) {
  return {
    id: message.id,
    role: message.role,
    text: message.text || "",
    timestamp: message.timestamp || new Date().toISOString(),
    sources: message.sources || [],
    warnings: message.warnings || [],
    streaming: Boolean(message.streaming),
  };
}

function renderConversationList() {
  dom.conversationList.replaceChildren();

  if (!state.conversations.length) {
    const placeholder = document.createElement("p");
    placeholder.className = "conversation-item__preview";
    placeholder.textContent = "No chats yet. Start a new conversation to create one.";
    dom.conversationList.append(placeholder);
    return;
  }

  state.conversations.forEach((conversation) => {
    const fragment = dom.conversationTemplate.content.cloneNode(true);
    const button = fragment.querySelector(".conversation-item");
    button.dataset.sessionId = conversation.session_id;
    if (conversation.session_id === state.currentSessionId) {
      button.classList.add("is-active");
    }
    button.querySelector(".conversation-item__title").textContent = conversation.title;
    button.querySelector(".conversation-item__meta").textContent =
      `${formatTimestamp(conversation.updated_at)} • ${conversation.model || "unknown"} • ${conversation.profile}`;
    button.querySelector(".conversation-item__preview").textContent = conversation.preview;
    dom.conversationList.append(fragment);
  });
}

function renderMessages() {
  dom.messages.replaceChildren();
  state.messages.forEach((message) => {
    dom.messages.append(createMessageElement(message));
  });
  renderTitle();
  toggleEmptyState();
}

function createMessageElement(message) {
  const fragment = dom.messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  article.dataset.messageId = message.id;
  hydrateMessageElement(article, message);
  return article;
}

function hydrateMessageElement(element, message) {
  element.className = `message ${message.role}${message.streaming ? " is-streaming" : ""}`;
  element.querySelector(".message__meta").textContent =
    `${message.role === "user" ? "You" : "Assistant"} • ${formatTimestamp(message.timestamp)}`;
  element.querySelector(".message__text").textContent =
    message.text || (message.streaming ? "Working on it…" : "");

  const details = element.querySelector(".message__details");
  details.replaceChildren();

  if (message.sources && message.sources.length) {
    const detailsBlock = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${message.sources.length})`;
    detailsBlock.append(summary);

    const list = document.createElement("ul");
    message.sources.forEach((source) => {
      const item = document.createElement("li");
      item.className = "message-source";
      item.textContent = `${source.doc_id} • ${source.category} • ${source.source} • score=${source.score}`;
      list.append(item);
    });
    detailsBlock.append(list);
    details.append(detailsBlock);
  }

  if (message.warnings && message.warnings.length) {
    const warningsBlock = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Warnings (${message.warnings.length})`;
    warningsBlock.append(summary);

    const list = document.createElement("ul");
    message.warnings.forEach((warning) => {
      const item = document.createElement("li");
      item.className = "message-warning";
      item.textContent = warning;
      list.append(item);
    });
    warningsBlock.append(list);
    details.append(warningsBlock);
  }
}

function toggleEmptyState() {
  const hasMessages = state.messages.length > 0;
  dom.emptyState.classList.toggle("is-hidden", hasMessages);
}

function renderTitle() {
  dom.chatTitle.textContent = state.currentTitle || "New Chat";
}

function setStatus(text) {
  if (!text) {
    dom.statusPill.hidden = true;
    dom.statusPill.textContent = "";
    return;
  }
  dom.statusPill.hidden = false;
  dom.statusPill.textContent = text;
}

function autoResizeTextarea() {
  dom.messageInput.style.height = "0px";
  dom.messageInput.style.height = `${Math.min(dom.messageInput.scrollHeight, 180)}px`;
}

function scrollMessagesToBottom(smooth) {
  dom.messages.scrollTo({
    top: dom.messages.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
}

function upsertConversationSummary(summary) {
  const existingIndex = state.conversations.findIndex(
    (item) => item.session_id === summary.session_id,
  );
  if (existingIndex === -1) {
    state.conversations.unshift(summary);
    return;
  }
  state.conversations.splice(existingIndex, 1, {
    ...state.conversations[existingIndex],
    ...summary,
  });
}

function truncateText(text, limit) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (compact.length <= limit) {
    return compact;
  }
  return `${compact.slice(0, limit - 1).trimEnd()}…`;
}

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return "Unknown time";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function cssEscape(value) {
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(value);
  }
  return String(value).replace(/"/g, '\\"');
}

function makeClientId(prefix) {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return `${prefix}-${window.crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function findOption(selectElement, value) {
  return Array.from(selectElement.options).find((option) => option.value === value);
}
