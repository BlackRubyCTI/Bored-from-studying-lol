/* Hyper-Cluster Omega — Mission Control.
 *
 * No framework, no build step: the phone gets a PWA that talks to the
 * head node through the dashboard proxy.  Everything is local storage, an
 * EventSource and the Web Speech API.
 */
(function () {
  "use strict";

  var STORE = "hyper-cluster-omega";
  var state = {
    head: localStorage.getItem(STORE + ".head") || window.location.origin,
    key: localStorage.getItem(STORE + ".key") || "",
    admin: localStorage.getItem(STORE + ".admin") || "",
    voice: localStorage.getItem(STORE + ".voice") || "",
    council: localStorage.getItem(STORE + ".council") === "1",
    autopilot: localStorage.getItem(STORE + ".autopilot") === "1",
    rag: localStorage.getItem(STORE + ".rag") === "1",
    speak: localStorage.getItem(STORE + ".speak") === "1",
    voices: [],
    recognition: null,
    listening: false,
    streaming: false,
    // Capability truth, refreshed from /v1/status on every telemetry poll.
    answersAreReal: false,
    backends: []
  };

  // ── helpers ──────────────────────────────────────────────────────────
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function save() {
    localStorage.setItem(STORE + ".head", state.head);
    localStorage.setItem(STORE + ".key", state.key);
    localStorage.setItem(STORE + ".admin", state.admin);
    localStorage.setItem(STORE + ".voice", state.voice);
    localStorage.setItem(STORE + ".council", state.council ? "1" : "0");
    localStorage.setItem(STORE + ".autopilot", state.autopilot ? "1" : "0");
    localStorage.setItem(STORE + ".rag", state.rag ? "1" : "0");
    localStorage.setItem(STORE + ".speak", state.speak ? "1" : "0");
  }
  function status(text) { $("status-line").textContent = text || ""; }

  function endpoint(path) {
    var base = (state.head || "").trim().replace(/\/+$/, "");
    if (base && base !== window.location.origin) {
      return base + path;
    }
    return "/api" + path;
  }

  function headers(extra) {
    var out = { "content-type": "application/json" };
    if (state.key) out["x-api-key"] = state.key;
    if (state.admin || state.key) out["x-admin-token"] = state.admin || state.key;
    if (extra) Object.keys(extra).forEach(function (k) { out[k] = extra[k]; });
    return out;
  }

  function api(path, options) {
    var opts = options || {};
    opts.headers = headers(opts.headers);
    opts.method = opts.method || "GET";
    return fetch(endpoint(path), opts).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (body) {
          throw new Error(body && body.length ? body.slice(0, 200) : "HTTP " + response.status);
        });
      }
      return response.json();
    });
  }

  // ── the conversation ─────────────────────────────────────────────────
  var chat = $("chat");

  function scrollDown() {
    requestAnimationFrame(function () { chat.scrollTop = chat.scrollHeight; });
  }

  function addMessage(role, text, opts) {
    opts = opts || {};
    var empty = $("empty-state");
    if (empty) empty.remove();
    var wrap = el("div", "msg " + role);
    wrap.appendChild(el("div", "avatar", role === "user" ? "🧑" : "⚡"));
    var bubble = el("div", "bubble");
    if (opts.typing) {
      var dots = el("span", "typing");
      dots.appendChild(el("span")); dots.appendChild(el("span")); dots.appendChild(el("span"));
      bubble.appendChild(dots);
    } else {
      bubble.textContent = text || "";
    }
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollDown();
    return bubble;
  }

  function setBubble(bubble, text) { bubble.textContent = text; scrollDown(); }

  // ── streaming chat ───────────────────────────────────────────────────
  function send(text) {
    text = (text || "").trim();
    if (!text || state.streaming) return;
    addMessage("user", text);
    $("input").value = "";
    autoGrow();
    var bubble = addMessage("assistant", "", { typing: true });
    state.streaming = true;
    status("thinking…");

    var payload = {
      messages: [{ role: "user", content: text }],
      mode: state.council ? "council" : "auto",
      stream: true,
      max_tokens: 512
    };

    var answer = "";
    fetch(endpoint("/v1/chat/stream"), {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(payload)
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var kind = "";
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { finish(); return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var lines = buffer.split("\n");
          buffer = lines.pop();
          lines.forEach(handleLine);
          return pump();
        });
      }
      function handleLine(line) {
        if (line.indexOf("event:") === 0) { kind = line.slice(6).trim(); return; }
        if (line.indexOf("data:") !== 0) return;
        var raw = line.slice(5).trim();
        if (!raw || raw === "[DONE]") return;
        var data;
        try { data = JSON.parse(raw); } catch (err) { return; }
        if (kind === "token") {
          answer += data.text || "";
          setBubble(bubble, answer);
          status("streaming…");
        } else if (kind === "done") {
          if (data.text && !answer) { answer = data.text; setBubble(bubble, answer); }
          finish();
        } else if (kind === "deferred") {
          setBubble(bubble, data.message || "⏳ Request queued for replay (" + (data.request_id || "pending") + ").");
          finish();
        } else if (kind === "error") {
          setBubble(bubble, "⚠️ " + (data.message || data.error || "the cluster failed"));
          finish();
        }
      }
      function finish() {
        if (!state.streaming) return;
        state.streaming = false;
        if (bubble.classList.contains("bubble")) {
          var typing = bubble.querySelector(".typing");
          if (typing) typing.remove();
          if (!bubble.textContent) bubble.textContent = "(no answer)";
        }
        status("");
        if (state.speak && answer) speak(answer);
        scrollDown();
      }
      return pump();
    }).catch(function (err) {
      state.streaming = false;
      var typing = bubble.querySelector(".typing");
      if (typing) typing.remove();
      setBubble(bubble, "⚠️ " + err.message);
      status("");
    });
  }

  // ── voice ────────────────────────────────────────────────────────────
  function speak(text) {
    if (!("speechSynthesis" in window)) return;
    try {
      var utter = new SpeechSynthesisUtterance(text);
      if (state.voice) {
        state.voices.forEach(function (voice) {
          if (voice.name === state.voice) utter.voice = voice;
        });
      }
      utter.rate = 1.02;
      utter.pitch = 1.0;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utter);
    } catch (err) { /* the phone has no voice */ }
  }

  function loadVoices() {
    if (!("speechSynthesis" in window)) return;
    state.voices = window.speechSynthesis.getVoices() || [];
    var select = $("set-voice");
    if (!select) return;
    select.innerHTML = "";
    var blank = document.createElement("option");
    blank.value = ""; blank.textContent = "(browser default)";
    select.appendChild(blank);
    state.voices.forEach(function (voice) {
      var option = document.createElement("option");
      option.value = voice.name;
      option.textContent = voice.name + " (" + voice.lang + ")";
      select.appendChild(option);
    });
    select.value = state.voice;
  }

  function toggleListening() {
    var Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      status("this browser has no speech recognition — type instead");
      return;
    }
    if (state.listening) { stopListening(); return; }
    var recognition = new Recognition();
    recognition.lang = navigator.language || "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onresult = function (event) {
      var text = "";
      for (var i = event.resultIndex; i < event.results.length; i++) text += event.results[i][0].transcript;
      $("input").value = text;
      autoGrow();
    };
    recognition.onend = function () { stopListening(); };
    recognition.onerror = function (event) {
      status("voice: " + (event.error || "failed"));
      stopListening();
    };
    try {
      recognition.start();
    } catch (err) {
      status("voice: " + err.message);
      return;
    }
    state.recognition = recognition;
    state.listening = true;
    $("btn-mic").classList.add("listening");
    status("listening…");
  }

  function stopListening() {
    if (state.recognition) {
      try { state.recognition.stop(); } catch (err) { /* already stopped */ }
    }
    state.recognition = null;
    state.listening = false;
    $("btn-mic").classList.remove("listening");
  }

  // ── telemetry ────────────────────────────────────────────────────────
  function refreshTelemetry() {
    api("/v1/status").then(function (data) {
      $("m-workers").textContent = (data.healthy_workers || 0) + "/" + (data.worker_count || 0);
      $("m-vram").textContent = Math.round((data.free_vram_gb || 0)) + "/" + Math.round((data.total_vram_gb || 0));
      $("m-tps").textContent = Math.round(data.tokens_per_sec_aggregate || 0);
      $("m-model").textContent = (data.active_model || "—").split("/").pop();
      $("m-queue").textContent = data.queued_requests || 0;
      $("cluster-name").textContent = data.cluster_name || "omega";
      applyCapability(data.capability, data.answers_are_real, data.backends);
      var dot = $("health-dot");
      dot.className = "dot " + (data.healthy ? "dot-on" : "dot-warn");
      if (typeof data.autopilot === "boolean" && !state.streaming) {
        setSwitch($("sw-autopilot"), data.autopilot);
        state.autopilot = data.autopilot;
      }
    }).catch(function (err) {
      $("health-dot").className = "dot dot-off";
      status("cluster unreachable: " + err.message);
    });
  }

  // Capability truth: the badge must never imply model-grade answers when the
  // only backend behind the cluster is the deterministic simulator.
  function applyCapability(capability, answersAreReal, backends) {
    var label = (capability && capability.label) ||
      (answersAreReal ? "live model" : "simulator");
    var real = typeof answersAreReal === "boolean" ? answersAreReal :
      Boolean(capability && capability.answers_are_real);
    var node = $("m-capability");
    var text = $("m-capability-label");
    if (!node || !text) return;
    text.textContent = real ? "live model" : "simulator — not model output";
    node.className = "capability " + (real ? "capability-live" : "capability-simulator");
    node.title = (capability && capability.detail) ||
      (real ? "Real model inference is serving." : "Answers come from the deterministic simulator.");
    node.setAttribute("aria-label", text.textContent + (real ? "" : " (simulated answers)"));
    state.answersAreReal = real;
    state.backends = backends || (capability && capability.backends) || [];
  }

  function setSwitch(node, on) {
    node.setAttribute("aria-pressed", on ? "true" : "false");
  }

  // ── the RAG upload ───────────────────────────────────────────────────
  function upload(files) {
    if (!files || !files.length) return;
    var queue = Array.prototype.slice.call(files);
    function next() {
      if (!queue.length) { status(""); return; }
      var file = queue.shift();
      var form = new FormData();
      form.append("file", file, file.name);
      status("indexing " + file.name + "…");
      fetch(endpoint("/v1/memory/documents"), {
        method: "POST",
        headers: state.key ? { "x-api-key": state.key } : {},
        body: form
      }).then(function (response) {
        if (!response.ok) return response.text().then(function (body) { throw new Error(body.slice(0, 160)); });
        return response.json();
      }).then(function (body) {
        var doc = body.document || {};
        status("✓ " + file.name + " → " + (doc.chunks || 0) + " chunks");
        addMessage("system", "Indexed " + file.name + " (" + (doc.chunks || 0) + " chunks" +
          (body.pdf_pages ? ", " + body.pdf_pages + " pages" : "") + ").");
        next();
      }).catch(function (err) {
        status("upload failed: " + err.message);
        next();
      });
    }
    next();
  }

  // ── wiring ──────────────────────────────────────────────────────────
  function autoGrow() {
    var input = $("input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  function init() {
    $("set-head").value = state.head;
    $("set-key").value = state.key;
    $("set-admin").value = state.admin;
    setSwitch($("sw-council"), state.council);
    setSwitch($("sw-autopilot"), state.autopilot);
    setSwitch($("sw-rag"), state.rag);
    setSwitch($("sw-voice"), state.speak);

    $("btn-send").addEventListener("click", function () { send($("input").value); });
    $("input").addEventListener("input", autoGrow);
    $("input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        send($("input").value);
      }
    });
    $("btn-mic").addEventListener("click", toggleListening);

    $("sw-council").addEventListener("click", function () {
      state.council = !state.council;
      setSwitch($("sw-council"), state.council);
      save();
      status(state.council ? "council mode: a panel answers" : "council mode off");
    });

    $("sw-autopilot").addEventListener("click", function () {
      var next = !state.autopilot;
      state.autopilot = next;
      setSwitch($("sw-autopilot"), next);
      save();
      api("/v1/evolution/toggle", { method: "POST", body: JSON.stringify({ enabled: next }) })
        .then(function () { status("autopilot " + (next ? "on" : "off")); })
        .catch(function (err) {
          state.autopilot = !next;
          setSwitch($("sw-autopilot"), state.autopilot);
          status("autopilot switch failed: " + err.message);
        });
    });

    $("sw-rag").addEventListener("click", function () {
      state.rag = !state.rag;
      setSwitch($("sw-rag"), state.rag);
      save();
      $("dropzone").classList.toggle("active", state.rag);
      if (state.rag) $("file-input").click();
    });

    $("sw-voice").addEventListener("click", function () {
      state.speak = !state.speak;
      setSwitch($("sw-voice"), state.speak);
      save();
      status(state.speak ? "the answers are spoken" : "voice output off");
    });

    $("btn-attach").addEventListener("click", function () { $("file-input").click(); });
    $("file-input").addEventListener("change", function (event) {
      upload(event.target.files);
      event.target.value = "";
    });

    var drop = document.body;
    ["dragenter", "dragover"].forEach(function (name) {
      drop.addEventListener(name, function (event) {
        event.preventDefault();
        $("dropzone").classList.add("active");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      drop.addEventListener(name, function (event) {
        event.preventDefault();
        if (!state.rag) $("dropzone").classList.remove("active");
      });
    });
    drop.addEventListener("drop", function (event) {
      if (event.dataTransfer && event.dataTransfer.files.length) upload(event.dataTransfer.files);
    });

    $("btn-settings").addEventListener("click", function () {
      $("settings").classList.remove("hidden");
      $("settings-hint").textContent = "The dashboard talks to " + state.head;
    });
    $("btn-telemetry").addEventListener("click", refreshTelemetry);
    $("btn-save").addEventListener("click", function () {
      state.head = $("set-head").value.trim() || window.location.origin;
      state.key = $("set-key").value.trim();
      state.admin = $("set-admin").value.trim();
      state.voice = $("set-voice").value;
      save();
      $("settings").classList.add("hidden");
      refreshTelemetry();
    });
    $("btn-logout").addEventListener("click", function () {
      state.key = "";
      state.admin = "";
      $("set-key").value = "";
      $("set-admin").value = "";
      save();
      status("the keys were cleared");
    });

    document.querySelectorAll("#suggestions .chip").forEach(function (chip) {
      chip.addEventListener("click", function () { send(chip.dataset.prompt); });
    });

    if ("speechSynthesis" in window) {
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    refreshTelemetry();
    setInterval(refreshTelemetry, 3000);

    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(function () { /* offline is optional */ });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
