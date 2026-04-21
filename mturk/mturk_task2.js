<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<crowd-form answer-format="flatten-objects" style="padding-bottom: 40px;">

  <!-- Hidden source fields from CSV -->
  <div id="mturk-data" style="display:none;">
    <div id="instruction-src">${instruction}</div>
    <div id="scenario-src">${scenario_description}</div>
    <div id="question-src">${question_commitment}</div>
    <div id="reasoning-src">${reasoning_block}</div>
  </div>

  <!-- Instructions -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Instructions</h3>
    <div id="instruction-box" style="font-size: 15px; line-height: 1.6;"></div>

    <div style="margin-top: 14px; padding: 12px; background: #f8fafc; border-left: 4px solid #2563eb; border-radius: 6px;">
      <div style="font-weight: 700; margin-bottom: 6px;">What you should do</div>
      <ol style="margin: 0; padding-left: 18px; line-height: 1.7;">
        <li>Read the full reasoning block.</li>
        <li>Select the earliest sentence where the person has clearly committed to a decision.</li>
        <li>If no sentence shows a clear commitment, select <b>No clear decision yet</b>.</li>
      </ol>
    </div>

    <div style="margin-top: 18px; color: #b22222; font-weight: bold; background: #fff0f0; padding: 12px; border-left: 4px solid #b22222; border-radius: 6px;">
      Some questions may have known answers. Low accuracy may affect payment.
    </div>
  </div>

  <!-- Scenario -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Scenario</h3>
    <div id="scenario-box" style="font-size: 15px; line-height: 1.6;"></div>
  </div>

  <!-- Reasoning block -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Full reasoning block</h3>
    <div id="reasoning-box" style="white-space: pre-wrap; line-height: 1.8; font-size: 15px; background: #f8f8f8; border: 1px solid #e8e8e8; border-radius: 8px; padding: 16px;"></div>
  </div>

  <!-- Single question -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto 24px auto; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-radius: 12px; background: #fff;">
    <h3 style="margin-top: 0;">Question</h3>
    <div id="question-box" style="font-size: 16px; font-weight: 600; line-height: 1.6; margin-bottom: 14px;"></div>

    <div style="font-size: 14px; color: #444; margin-bottom: 14px;">
      Choose the <b>first</b> sentence where the decision becomes clear. If the reasoning never clearly commits to a decision, choose <b>No clear decision yet</b>.
    </div>

    <div id="sentence-choice-container" style="display: flex; flex-direction: column; gap: 10px;"></div>

    <div id="validation-error" style="display:none; margin-top: 14px; color: #b22222; font-weight: bold; background: #fff0f0; padding: 12px; border-left: 4px solid #b22222; border-radius: 6px;">
      Please select one answer before submitting.
    </div>
  </div>

  <!-- Hidden outputs -->
  <input type="hidden" name="commitment_sentence_index" id="commitment_sentence_index" value="">
  <input type="hidden" name="commitment_sentence_text" id="commitment_sentence_text" value="">
  <input type="hidden" name="sentence_count" id="sentence_count" value="">
  <input type="hidden" name="task_id_echo" value="${task_id}">
  <input type="hidden" name="example_id_echo" value="${example_id}">
  <input type="hidden" name="environment_echo" value="${environment}">
  <input type="hidden" name="spike_sentence_idx_echo" value="${spike_sentence_idx}">

  <!-- Optional feedback -->
  <div style="width: 100%; max-width: 980px; margin: 0 auto;">
    <h4>Optional feedback</h4>
    <p>If anything was confusing or unclear, please let us know.</p>
    <textarea
      name="worker_feedback"
      rows="5"
      style="width: 100%; padding: 12px; font-size: 14px; border-radius: 6px; border: 1px solid #ccc; resize: vertical;"
      placeholder="Enter your feedback here..."></textarea>
  </div>

</crowd-form>

<script>
  function cleanText(text) {
    if (!text) return "";
    const trimmed = text.trim();
    if (
      trimmed === "nan" ||
      trimmed === "None" ||
      trimmed === "undefined" ||
      trimmed === "null"
    ) {
      return "";
    }
    return trimmed;
  }

  function normalizeWhitespace(text) {
    return cleanText(text).replace(/\s+/g, " ").trim();
  }

  function splitIntoSentences(text) {
    const normalized = normalizeWhitespace(text);
    if (!normalized) return [];

    const pieces = normalized.split(/(?<=[.!?])\s+(?=[A-Z0-9"'(\[])/);
    const sentences = pieces.map(s => s.trim()).filter(Boolean);

    if (sentences.length === 0) {
      return [normalized];
    }
    return sentences;
  }

  function fillStaticText() {
    document.getElementById("instruction-box").textContent =
      cleanText(document.getElementById("instruction-src").textContent);

    document.getElementById("scenario-box").textContent =
      cleanText(document.getElementById("scenario-src").textContent);

    const providedQuestion = cleanText(document.getElementById("question-src").textContent);
    document.getElementById("question-box").textContent =
      providedQuestion || "What is the earliest sentence where the person has clearly committed to a decision?";

    document.getElementById("reasoning-box").textContent =
      cleanText(document.getElementById("reasoning-src").textContent);
  }

  function highlightSelectedSentence(selectedIndex) {
    const nodes = document.querySelectorAll(".sentence-choice");
    nodes.forEach(node => {
      if (node.dataset.sentenceIndex === selectedIndex) {
        node.style.border = "2px solid #2563eb";
        node.style.background = "#eff6ff";
      } else {
        node.style.border = "1px solid #ddd";
        node.style.background = "#fafafa";
      }
    });
  }

  function buildSentenceChoices() {
    const container = document.getElementById("sentence-choice-container");
    container.innerHTML = "";

    const reasoning = cleanText(document.getElementById("reasoning-src").textContent);
    const sentences = splitIntoSentences(reasoning);

    document.getElementById("sentence_count").value = String(sentences.length);

    const noneId = "commitment_sentence_none";
    const noneWrapper = document.createElement("label");
    noneWrapper.setAttribute("for", noneId);
    noneWrapper.className = "sentence-choice";
    noneWrapper.dataset.sentenceIndex = "0";
    noneWrapper.style.display = "flex";
    noneWrapper.style.alignItems = "flex-start";
    noneWrapper.style.gap = "12px";
    noneWrapper.style.padding = "14px 16px";
    noneWrapper.style.border = "1px solid #ddd";
    noneWrapper.style.borderRadius = "10px";
    noneWrapper.style.background = "#fff8f0";
    noneWrapper.style.cursor = "pointer";

    const noneRadio = document.createElement("input");
    noneRadio.type = "radio";
    noneRadio.name = "commitment_sentence_choice";
    noneRadio.id = noneId;
    noneRadio.value = "0";
    noneRadio.dataset.sentenceText = "No clear decision yet";
    noneRadio.required = true;
    noneRadio.style.marginTop = "3px";

    noneRadio.addEventListener("change", function () {
      document.getElementById("commitment_sentence_index").value = "0";
      document.getElementById("commitment_sentence_text").value = "No clear decision yet";
      document.getElementById("validation-error").style.display = "none";
      highlightSelectedSentence("0");
    });

    const noneText = document.createElement("div");
    noneText.innerHTML = '<div style="font-size:15px; font-weight:600;">No clear decision yet</div><div style="font-size:13px; color:#555; margin-top:4px;">Use this only if no sentence clearly commits to a decision.</div>';

    noneWrapper.appendChild(noneRadio);
    noneWrapper.appendChild(noneText);
    container.appendChild(noneWrapper);

    sentences.forEach((sentence, idx) => {
      const sentenceNumber = idx + 1;
      const optionId = "commitment_sentence_" + sentenceNumber;

      const wrapper = document.createElement("label");
      wrapper.setAttribute("for", optionId);
      wrapper.className = "sentence-choice";
      wrapper.dataset.sentenceIndex = String(sentenceNumber);
      wrapper.style.display = "flex";
      wrapper.style.alignItems = "flex-start";
      wrapper.style.gap = "12px";
      wrapper.style.padding = "14px 16px";
      wrapper.style.border = "1px solid #ddd";
      wrapper.style.borderRadius = "10px";
      wrapper.style.background = "#fafafa";
      wrapper.style.cursor = "pointer";

      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "commitment_sentence_choice";
      radio.id = optionId;
      radio.value = String(sentenceNumber);
      radio.dataset.sentenceText = sentence;
      radio.style.marginTop = "3px";

      radio.addEventListener("change", function () {
        document.getElementById("commitment_sentence_index").value = String(sentenceNumber);
        document.getElementById("commitment_sentence_text").value = sentence;
        document.getElementById("validation-error").style.display = "none";
        highlightSelectedSentence(String(sentenceNumber));
      });

      const textWrap = document.createElement("div");
      textWrap.style.lineHeight = "1.6";
      textWrap.style.flex = "1";

      const num = document.createElement("div");
      num.textContent = "Sentence " + sentenceNumber;
      num.style.fontWeight = "700";
      num.style.fontSize = "13px";
      num.style.color = "#374151";
      num.style.marginBottom = "4px";

      const sent = document.createElement("div");
      sent.textContent = sentence;
      sent.style.fontSize = "15px";

      textWrap.appendChild(num);
      textWrap.appendChild(sent);

      wrapper.appendChild(radio);
      wrapper.appendChild(textWrap);
      container.appendChild(wrapper);
    });
  }

  function attachValidation() {
    const form = document.querySelector("crowd-form");
    form.addEventListener("submit", function (e) {
      const checked = document.querySelector('input[name="commitment_sentence_choice"]:checked');
      if (!checked) {
        e.preventDefault();
        document.getElementById("validation-error").style.display = "block";
      }
    });
  }

  fillStaticText();
  buildSentenceChoices();
  attachValidation();
</script>
